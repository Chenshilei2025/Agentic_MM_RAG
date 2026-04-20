"""Orchestrator — Workflow engine. NO LLM. NO reasoning.

Responsibilities:
  * spawn / kill sub-agents
  * manage execution queue
  * parallelize sub-agent steps
  * route commands
  * maintain lifecycle & system state
  * enforce timeouts and retries
"""
from typing import Any, Dict
from tools.registry import ToolRegistry
from memory.state_manager import StateManager
from memory.evidence_pool import EvidencePool
from cli.schemas.commands import validate_command, CommandError
from orchestrator.runtime import (
    TaskQueue, WorkerPool, AgentRecord, Task,
    PENDING, RUNNING, DONE, KILLED, FAILED,
    new_task_id, new_agent_id,
)
from utils.logger import get_logger

log = get_logger("orchestrator")
STAGES = ("intent", "summary", "full")


def _heuristic_reflect_verdict(board, agreements, subtasks):
    """Fallback verdict when reflect_skill is unavailable (no provider,
    scripted provider, provider failure). Not as smart as the real skill
    but gives DA *something* actionable. Used in tests too.
    """
    conflicts = [{"aspect": a["aspect"],
                  "agent_ids": a.get("agent_ids", []),
                  "reason": "disagree state detected"}
                 for a in (agreements or [])
                 if a.get("agreement_state") == "disagree"]
    # Gaps: any important (>=0.7) subtask whose aspect has no summary row
    rows_by_aspect = {r.get("aspect"): r for r in (board or [])
                      if r.get("aspect")}
    gaps = []
    for s in (subtasks or []):
        if getattr(s, "importance", 0.0) < 0.7:
            continue
        row = rows_by_aspect.get(s.aspect)
        if row is None or row.get("version", 0) == 0:
            gaps.append({"aspect": s.aspect,
                         "modality": (s.modalities[0] if s.modalities else ""),
                         "reason": "no summary yet"})
        elif row.get("coverage", {}).get("covered", 0.0) < 0.5:
            gaps.append({"aspect": s.aspect,
                         "modality": (s.modalities[0] if s.modalities else ""),
                         "reason": "thin coverage"})
    # Decide action
    if conflicts:
        action = "ESCALATE"
    elif gaps:
        # If gap aspect has NO row at all → REPLAN; else ESCALATE
        action = ("REPLAN"
                  if any(not rows_by_aspect.get(g["aspect"]) for g in gaps)
                  else "ESCALATE")
    elif (board and all(r.get("coverage", {}).get("covered", 0) >= 0.7
                        for r in board if r.get("version", 0) > 0)):
        action = "ANSWER"
    else:
        action = "WAIT"
    return {
        "can_answer": action == "ANSWER",
        "conflicts": conflicts,
        "gaps": gaps,
        "recommended_action": action,
        "escalation_targets": [c["agent_ids"][0] for c in conflicts
                               if c.get("agent_ids")],
        "source": "heuristic",
    }


class Orchestrator:
    def __init__(self, decision_agent, factory, store,
                 max_steps: int = 40, max_workers: int = 4,
                 step_timeout: float = 30.0, ablation=None,
                 asset_resolver=None,
                 max_tokens: int = 100_000,
                 enable_decompose: bool = False):
        if ablation is None:
            from config.ablation import AblationConfig
            ablation = AblationConfig()
        self.ablation = ablation
        if not ablation.enable_parallelism:
            max_workers = 1
        self.decision_agent = decision_agent
        self.factory = factory
        self.store = store
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        # Off by default: decompose consumes one extra DA call and only helps
        # if the DA is a real LLM. Tests with scripted/mock providers must
        # leave this off. Live runs turn it on explicitly.
        self.enable_decompose = enable_decompose
        self.queue = TaskQueue()
        self.workers = WorkerPool(max_workers, step_timeout)
        self.agents: Dict[str, AgentRecord] = {}
        self._instances: Dict[str, Any] = {}
        if asset_resolver is None:
            from utils.asset_resolver import LocalFileResolver
            asset_resolver = LocalFileResolver()
        self.asset_resolver = asset_resolver

    # ---------- PUBLIC ----------
    def run(self, query: str) -> Dict[str, Any]:
        pool = EvidencePool(
            bypass_tier3_gate=not self.ablation.enable_tier3_gate)
        state = StateManager(query=query)
        state.pool = pool
        state.agent_records = self.agents
        state.max_steps = self.max_steps
        # Budget — soft gate on SKETCH/FULL requests via VoI layer.
        from orchestrator.runtime import Budget
        state.budget = Budget(
            max_tokens=getattr(self, "max_tokens", 100_000),
            used_tokens=0)

        # Pre-run: optional structured subtask decomposition. Disabled by
        # default; live demos enable it via Orchestrator(enable_decompose=True).
        if self.enable_decompose:
            self._pre_run_decompose(query, state)

        try:
            # Snapshot provider usage counters at the start so we can measure
            # per-loop deltas (works even when multiple sub-agents share
            # the same provider object).
            def _total_tokens():
                provs = {getattr(self.decision_agent, "_provider", None)}
                for a in self._instances.values():
                    provs.add(getattr(a, "_provider", None))
                total = 0
                for p in provs:
                    if p is None: continue
                    total += int(getattr(p, "total_input_tokens", 0) or 0)
                    total += int(getattr(p, "total_output_tokens", 0) or 0)
                return total
            tokens_at_start = _total_tokens()

            while not state.is_done() and state.step_count < self.max_steps:
                state.step_count += 1
                log.info(f"===== step {state.step_count} =====")

                # Round 4 — record info-gain BEFORE the DA step, so the
                # tracker reflects the state the DA sees. Includes any
                # evidence written by sub-agents in the prior turn.
                state.info_gain_tracker.record(
                    turn=state.step_count,
                    max_conf=pool.max_confidence(),
                    mean_coverage=pool.coverage())
                if state.info_gain_tracker.is_saturated():
                    log.info(f"[info-gain] SATURATED after "
                             f"{state.step_count} turns — new retrieval "
                             f"requests will be DENIED; DA should STOP")

                # Round 4 — check loop progress BEFORE DA step to detect
                # stagnation early. Force stop if no progress for multiple
                # turns AND approaching max_steps or redundant commands.
                if state.loop_progress_tracker.should_force_stop(
                    state.step_count, self.max_steps):
                    rec_action = state.loop_progress_tracker.get_recommended_action()
                    log.warning(f"[loop-progress] Force stop at step "
                              f"{state.step_count}: {rec_action}")
                    state.finalize(
                        answer=pool.synthesize_answer(
                            state.conflict_resolutions,
                            ablation=self.ablation),
                        confidence=pool.max_confidence(),
                        reason=f"loop_progress_force_stop:{rec_action}")
                    break

                raw = self.decision_agent.step(state)
                pool.consume_deltas()
                try:
                    cmd = validate_command(raw)
                except CommandError as e:
                    log.error(f"bad command: {e}")
                    continue
                log.info(f"[cmd] {cmd}")

                self._dispatch(cmd, state, pool)
                if state.is_done():
                    break

                self._run_queued_steps(state, pool)

                # Sync FAILED status from AgentRecord to the pool snapshot
                for aid, rec in self.agents.items():
                    if rec.status == FAILED:
                        pool.set_agent_status(aid, "failed")

                # Update the soft token budget. We re-total across providers
                # each turn rather than deltaing to avoid missing sub-agent
                # providers that spun up during dispatch.
                if state.budget is not None:
                    state.budget.used_tokens = _total_tokens() - tokens_at_start

                # Round 4 — record loop progress AFTER sub-agent steps complete,
                # so we capture the full state of this turn (including any
                # evidence written by sub-agents).
                from utils.loop_progress import build_turn_snapshot
                commands_issued = [cmd.get("command", "") for cmd in [cmd]]
                snapshot = build_turn_snapshot(state, pool, commands_issued)
                signal = state.loop_progress_tracker.record_turn(snapshot)
                log.info(f"[loop-progress] {signal.value} "
                         f"(aspects={len(snapshot.covered_aspects)}, "
                         f"conflicts={len(snapshot.conflict_aspects)}, "
                         f"quality={snapshot.evidence_quality:.2f})")
                # Add progress summary to trace for analysis
                state.trace.append({
                    "type": "loop_progress",
                    "step": state.step_count,
                    "signal": signal.value,
                    "progress": state.loop_progress_tracker.get_progress_summary()
                })
        finally:
            self.workers.shutdown()
            # Expose final snapshot for benchmark harness / external
            # observers. Always set, even if the main loop raised.
            try:
                self.last_status_board = pool.status_board(include_terminated=True)
                self.last_pool_trace = pool.trace()
                self.last_state_trace = list(state.trace)
            except Exception:
                self.last_status_board = []
                self.last_pool_trace = []
                self.last_state_trace = []

        return state.final or {"answer": None,
                               "confidence": pool.max_confidence(),
                               "reason": "max_steps_exhausted"}

    # ---------- COMMAND DISPATCH ----------
    def _dispatch(self, cmd, state, pool):
        name, args = cmd["command"], cmd["arguments"]
        if name == "SPAWN_AGENT":
            self._spawn(args, state, pool)
        elif name == "SPAWN_AGENTS":
            for spec in args["specs"]:
                self._spawn(spec, state, pool)
            log.info(f"[spawn-batch] {len(args['specs'])} agents")
        elif name == "KILL_AGENT":
            rec = self._require(args["agent_id"])
            rec.status = KILLED
            pool.set_agent_status(rec.agent_id, "killed")
            log.info(f"[kill] {args['agent_id']}")
        elif name == "SWITCH_MODALITY":
            rec = self._require(args["agent_id"])
            rec.modality = args["modality"]
            log.info(f"[modality] {rec.agent_id} -> {rec.modality}")
        elif name == "CONTINUE_RETRIEVAL":
            # Round 4 — saturation hard rule applies here too.
            tr = getattr(state, "info_gain_tracker", None)
            if tr is not None and tr.is_saturated():
                log.warning(f"[continue-rejected] info-gain saturated; "
                            f"re-retrieval denied. DA must STOP.")
                return
            # DA decides WHETHER to re-retrieve, and provides SEMANTIC hint
            # (what to focus on / reformulated query). Orchestrator decides
            # HOW: computes exclude_ids so sub-agent doesn't re-see the same
            # candidates, checks budget, re-queues with fresh feedback.
            rec = self._require(args["agent_id"])
            if rec.status in (KILLED, FAILED):
                log.warning(f"cannot continue {rec.status} agent {rec.agent_id}")
                return
            hint = args.get("hint", "")
            # Orchestrator's "how": build the exclude set from what this
            # agent already retrieved + saw, so re-retrieval is productive.
            already_seen = {c.get("id") for c in rec.last_retrieval
                            if c.get("id")}
            feedback_parts = []
            if hint:
                feedback_parts.append(f"DA hint: {hint}")
            if already_seen:
                feedback_parts.append(
                    f"exclude_ids (already retrieved): "
                    f"{sorted(already_seen)[:20]}")
            feedback_parts.append(
                "You may call retrieval ONCE more with a refined query "
                "and exclude_ids argument.")
            state.push_feedback(rec.agent_id, "\n".join(feedback_parts))
            # Re-activate: bump retry budget so the guard doesn't block it.
            rec.status = PENDING
            rec.retries_left = max(rec.retries_left, 2)
            # Re-arm retrieval cap for this round (orchestrator's how)
            rec.continue_round = getattr(rec, "continue_round", 0) + 1
            self.queue.push(Task(new_task_id(), rec.agent_id, {"kind": "step"}))
            log.info(f"[continue] {rec.agent_id} round={rec.continue_round}  "
                     f"exclude_n={len(already_seen)}")
        elif name == "REQUEST_EVIDENCE_SKETCH":
            rec = self._require(args["agent_id"])
            if rec.status in (KILLED, FAILED):
                log.warning(f"cannot request sketch from {rec.status} "
                            f"agent {rec.agent_id}")
                return
            if pool.is_authorized_for_sketch(rec.agent_id):
                log.info(f"[authorize-sketch] {rec.agent_id} already pending")
                return
            # --- VoI gating ---
            from orchestrator.voi_gating import gate_request
            board = pool.status_board(include_terminated=True)
            snap_row = next((r for r in board
                             if r["agent_id"] == rec.agent_id), None)
            if snap_row is None:
                log.warning(f"[sketch-deny] {rec.agent_id} no snapshot")
                return
            subtask = self._find_subtask_for(rec, state)
            retry_count = self._voi_retry_count(rec.agent_id, "sketch", state)
            decision = gate_request(snap_row, board, subtask, state.budget,
                                    stage="sketch", retry_count=retry_count,
                                    info_gain_saturated=state.info_gain_tracker.is_saturated())
            state.voi_decisions.append(decision.to_trace())
            if not decision.allow:
                log.info(f"[voi-deny] sketch {rec.agent_id} "
                         f"reason={decision.reason} voi={decision.voi:.3f}")
                state.push_feedback(rec.agent_id,
                    f"DA's sketch request was blocked by VoI "
                    f"(reason={decision.reason}). Reply with STOP or "
                    f"try a different action.")
                return
            log.info(f"[voi-allow] sketch {rec.agent_id} "
                     f"reason={decision.reason}")
            pool.authorize_sketch(rec.agent_id)
            rec.status = PENDING
            rec.target_stage = "summary"
            state.push_feedback(rec.agent_id,
                "DA requested an EVIDENCE SKETCH (Tier-2). Pick 2-5 of "
                "the most relevant candidates from your retrieval, extract "
                "ONE key sentence from each that directly supports your "
                "finding, and call write_evidence(stage=\"sketch\", "
                "payload={\"key_sentences\":[{\"id\":..., \"text\":..., "
                "\"relevance\":0-1}, ...]}). Do NOT call retrieval again.")
            self.queue.push(Task(new_task_id(), rec.agent_id, {"kind": "step"}))
            log.info(f"[authorize-sketch] {rec.agent_id}")
        elif name == "INSPECT_EVIDENCE":
            # DA wants to see raw assets for specific ids. Resolve, stash on
            # state.pending_inspect, consumed on DA's NEXT prompt build.
            if state.inspect_count >= 3:
                log.warning(f"[inspect] run-level cap (3) reached, rejecting")
                return
            ids = args.get("ids") or []
            if not isinstance(ids, list) or not ids:
                log.warning("[inspect] no ids provided")
                return
            ids = ids[:5]  # per-call cap
            # Pool all agents' last_retrieval to find candidates
            wanted = set(str(i) for i in ids)
            found: List[Dict[str, Any]] = []
            for r in self.agents.values():
                for c in r.last_retrieval or []:
                    cid = str(c.get("id", ""))
                    if cid in wanted:
                        found.append(c)
                        wanted.discard(cid)
            missing = sorted(wanted)
            try:
                blocks = self.asset_resolver.resolve_many(
                    found,
                    header=(f"=== INSPECTED EVIDENCE (turn {state.step_count}) "
                            f"===\nDA requested: {ids}. "
                            f"Reason: {args.get('reason','(none)')}. "
                            + (f"Missing ids: {missing}. " if missing else "")
                            + "Review these carefully before acting."))
            except Exception as e:
                log.error(f"[inspect-fail] {e}")
                return
            state.pending_inspect.append({
                "turn": state.step_count,
                "ids": ids,
                "blocks": blocks,
            })
            state.inspect_count += 1
            log.info(f"[inspect] resolved {len(found)}/{len(ids)} "
                     f"(run-count={state.inspect_count}/3)")
        elif name == "REQUEST_FULL_EVIDENCE":
            rec = self._require(args["agent_id"])
            if rec.status in (KILLED, FAILED):
                log.warning(f"cannot request full from {rec.status} agent "
                            f"{rec.agent_id}")
                return
            if pool.is_authorized_for_full(rec.agent_id):
                log.info(f"[authorize-full] {rec.agent_id} already pending")
                return
            # --- VoI gating: same policy as sketch but for the more expensive
            # Tier-3. Retry override allows DA to push through on the 2nd ask.
            from orchestrator.voi_gating import gate_request
            board = pool.status_board(include_terminated=True)
            snap_row = next((r for r in board
                             if r["agent_id"] == rec.agent_id), None)
            if snap_row is None:
                log.warning(f"[full-deny] {rec.agent_id} no snapshot")
                return
            subtask = self._find_subtask_for(rec, state)
            retry_count = self._voi_retry_count(rec.agent_id, "full", state)
            decision = gate_request(snap_row, board, subtask, state.budget,
                                    stage="full", retry_count=retry_count,
                                    info_gain_saturated=state.info_gain_tracker.is_saturated())
            state.voi_decisions.append(decision.to_trace())
            if not decision.allow:
                log.info(f"[voi-deny] full {rec.agent_id} "
                         f"reason={decision.reason} voi={decision.voi:.3f}")
                state.push_feedback(rec.agent_id,
                    f"DA's full-evidence request was blocked by VoI "
                    f"(reason={decision.reason}). If you still need it, "
                    f"request again — the retry will be approved.")
                return
            log.info(f"[voi-allow] full {rec.agent_id} "
                     f"reason={decision.reason}")
            pool.authorize_full(rec.agent_id)
            rec.target_stage = "full"
            rec.status = PENDING
            state.push_feedback(rec.agent_id,
                                "produce Tier-3 full evidence (authorized)")
            self.queue.push(Task(new_task_id(), rec.agent_id, {"kind": "step"}))
            log.info(f"[authorize-full] {rec.agent_id}")
        elif name == "READ_ARCHIVE":
            # DA pulls a previously-written Tier-3 archive into the next
            # prompt. Pulls the LATEST archive entry only (sub-agent may have
            # written multiple times; we surface the freshest).
            aid = args["agent_id"]
            entries = pool.get_archive(aid)
            if not entries:
                log.warning(f"[read-archive] {aid} no archive available")
                state.push_feedback(aid, "(no archive entries to read)")
                return
            latest = entries[-1]
            state.pending_archive.append({
                "agent_id": aid,
                "content": latest.get("content", ""),
                "sources": list(latest.get("sources", [])),
                "reason": args.get("reason", ""),
            })
            log.info(f"[read-archive] {aid} queued "
                     f"({len(latest.get('content',''))} chars)")
        elif name == "REQUEST_CURATED_EVIDENCE":
            # Unified Tier-2 disclosure with cross-modal evidence selection.
            # Uses CrossModalSelector to prioritize conflict + important-low-conf.
            aid = args["agent_id"]
            ids = args.get("ids") or None   # None = agent-level top-K
            with_raw = bool(args.get("with_raw", False))
            if aid not in self.agents:
                log.warning(f"[curated] unknown agent {aid}")
                return
            # Cannot request from a failed/killed agent
            a_status = self.agents[aid].status
            if a_status in (KILLED, FAILED):
                log.warning(f"[curated] cannot request from "
                            f"{a_status} agent {aid}")
                return
            # VoI gate — reuse existing gate with tier-tagged stage name
            from orchestrator.voi_gating import gate_request
            stage = "curated_raw" if with_raw else "curated_light"
            retry_key = f"{aid}:{stage}"
            retry_count = getattr(state, "retry_counts", {}).get(retry_key, 0)
            board = pool.status_board()
            row = next((r for r in board if r["agent_id"] == aid), None)
            if row is None:
                log.warning(f"[curated] no status row for {aid}")
                return
            # Find matching subtask for the agent (may be None)
            subtask = None
            for s in (state.subtasks or []):
                if getattr(s, "aspect", None) == getattr(self.agents[aid],
                                                         "aspect", None):
                    subtask = s
                    break
            decision = gate_request(
                snap_row=row,
                status_board=board,
                subtask=subtask,
                budget=getattr(state, "budget", None),
                stage=stage,
                retry_count=retry_count,
                info_gain_saturated=state.info_gain_tracker.is_saturated())
            if not hasattr(state, "voi_decisions"):
                state.voi_decisions = []
            state.voi_decisions.append({
                "turn": state.step_count, "agent_id": aid,
                "stage": stage, "decision": decision.to_trace()})
            if not decision.allow:
                log.info(f"[curated] DENY {aid} stage={stage} "
                         f"reason={decision.reason}")
                return

            # === Cross-modal evidence selection ===
            # Store curated candidates on agent for sub-agent to use when writing
            from utils.cross_modal_selector import (CrossModalSelector,
                                                   select_curated_evidence)

            # Identify conflict aspects for prioritization
            conflict_aspects = CrossModalSelector.identify_conflict_aspects(
                pool.aspect_agreements()
            )

            # Check if this is an important + low-confidence subtask
            is_important = subtask is not None and subtask.importance >= 0.7
            is_low_confidence = CrossModalSelector.is_important_low_confidence(row)

            # Get agent's retrieved candidates
            retrieved = getattr(self.agents[aid], "last_retrieved", [])
            if not retrieved:
                log.warning(f"[curated] agent {aid} has no retrieved candidates")
                # Still authorize so sub-agent can respond with empty curated
                pool.authorize_curated(aid, with_raw=with_raw, ids=ids)
                if not hasattr(state, "retry_counts"):
                    state.retry_counts = {}
                state.retry_counts[retry_key] = retry_count + 1
                log.info(f"[curated] AUTH {aid} stage={stage} (no candidates)")
                return

            # Select high-value candidates using CrossModalSelector
            selector = CrossModalSelector(max_per_agent=10, conflict_boost=2.0)
            curated = selector.select_curated(
                agent_id=aid,
                retrieved_candidates=retrieved,
                status_snapshot=row,
                conflict_aspects=conflict_aspects,
                is_important=is_important,
                is_low_confidence=is_low_confidence
            )

            # Store curated candidates on agent for sub-agent to use
            self.agents[aid]._curated_candidates = curated

            # Authorize the sub-agent to write curated evidence
            pool.authorize_curated(aid, with_raw=with_raw, ids=ids)

            # Send feedback to sub-agent so it knows to write curated
            raw_flag = "with_raw" if with_raw else "summary_only"
            state.push_feedback(aid, f"produce Tier-2 curated evidence ({raw_flag}, authorized)")

            if not hasattr(state, "retry_counts"):
                state.retry_counts = {}
            state.retry_counts[retry_key] = retry_count + 1

            log.info(f"[curated] AUTH {aid} stage={stage} n_curated={len(curated)} "
                     f"conflict_aspects={conflict_aspects}")
        elif name == "REFLECT":
            # DA pauses action and invokes the REFLECT skill. Verdict is
            # stored on state so the next turn's DA prompt surfaces it.
            from prompts.skills import reflect_skill
            provider = getattr(self.decision_agent, "_provider", None)
            subtasks_snapshot = list(state.subtasks or [])
            board = pool.status_board()
            agreements = pool.aspect_agreements()
            verdict = reflect_skill.run(
                query=state.query,
                status_board=board,
                aspect_agreements=agreements,
                subtasks=subtasks_snapshot,
                provider=provider)
            # Round 6 — approx token accounting for per-tier breakdown.
            if state.budget is not None and verdict is not None:
                import json as _json
                approx = len(_json.dumps(board, default=str)) // 3 + 500
                state.budget.tier_breakdown["skill_reflect"] = (
                    state.budget.tier_breakdown.get("skill_reflect", 0) + approx)
            if verdict is None:
                log.warning("[reflect] skill returned None; "
                            "main loop falls back to heuristic decision")
                # Fallback: synthesize a minimal verdict from heuristics so
                # DA always sees SOMETHING on the next turn.
                verdict = _heuristic_reflect_verdict(board, agreements,
                                                     subtasks_snapshot)
            verdict["turn"] = state.step_count
            verdict["trigger"] = args.get("reason", "")
            state.pending_reflect_verdict = verdict
            state.reflect_verdicts.append(verdict)
            log.info(f"[reflect] verdict: "
                     f"action={verdict.get('recommended_action')} "
                     f"can_answer={verdict.get('can_answer')} "
                     f"conflicts={len(verdict.get('conflicts', []))} "
                     f"gaps={len(verdict.get('gaps', []))}")
        elif name == "REPLAN":
            # DA invokes REPLAN skill → patch → dispatched through existing
            # EXTEND_SUBTASKS / REVISE_SUBTASK / ABORT_AND_RESPAWN handlers.
            # All existing guardrails apply (cap 10, abort-once, etc).
            from prompts.skills import replan_skill
            provider = getattr(self.decision_agent, "_provider", None)
            last_verdict = state.pending_reflect_verdict or (
                state.reflect_verdicts[-1] if state.reflect_verdicts else {})
            board = pool.status_board()
            agreements = pool.aspect_agreements()
            patch = replan_skill.run(
                query=state.query,
                subtasks=list(state.subtasks or []),
                reflect_verdict=last_verdict,
                status_board=board,
                aspect_agreements=agreements,
                provider=provider)
            if state.budget is not None and patch is not None:
                import json as _json
                approx = len(_json.dumps(board, default=str)) // 3 + 600
                state.budget.tier_breakdown["skill_replan"] = (
                    state.budget.tier_breakdown.get("skill_replan", 0) + approx)
            if patch is None:
                log.warning("[replan] skill returned None; no patch applied")
                state.replan_traces.append({
                    "turn": state.step_count,
                    "patch": None, "applied": [],
                    "reason": "skill_unavailable"})
                return
            applied: List[str] = []
            # Execute each piece via existing dispatch paths
            for ext in (patch.get("extend") or [])[:5]:
                if not isinstance(ext, dict): continue
                self._dispatch(
                    {"command": "EXTEND_SUBTASKS",
                     "arguments": {"subtasks": [ext]}},
                    state, pool)
                applied.append(f"extend:{ext.get('aspect')}")
            for rev in (patch.get("revise") or [])[:5]:
                if not isinstance(rev, dict) or not rev.get("id"): continue
                self._dispatch(
                    {"command": "REVISE_SUBTASK",
                     "arguments": {k: v for k, v in rev.items()
                                   if v is not None}},
                    state, pool)
                applied.append(f"revise:{rev['id']}")
            for abort in (patch.get("abort_respawn") or [])[:3]:
                if not isinstance(abort, dict) or not abort.get("agent_id"):
                    continue
                self._dispatch(
                    {"command": "ABORT_AND_RESPAWN",
                     "arguments": {k: v for k, v in abort.items()
                                   if v is not None}},
                    state, pool)
                applied.append(f"abort:{abort['agent_id']}")
            state.replan_traces.append({
                "turn": state.step_count,
                "patch": patch, "applied": applied,
                "rationale": patch.get("rationale", "")})
            log.info(f"[replan] applied {len(applied)} ops: {applied}")
        elif name == "EXTEND_SUBTASKS":
            # DA discovered a missing aspect mid-run. Append to state.subtasks.
            # Hard cap: total subtasks capped at 10 to prevent DA from
            # decomposing indefinitely.
            MAX_SUBTASKS = 10
            from orchestrator.runtime import Subtask
            VALID_MODS = {"doc_text", "doc_visual",
                          "video_text", "video_visual"}
            new_specs = args.get("subtasks") or []
            existing_keys = {(getattr(s, "aspect", None),
                              tuple(sorted(getattr(s, "modalities", []) or [])))
                             for s in (state.subtasks or [])}
            added = 0
            for i, item in enumerate(new_specs[:5]):
                if len(state.subtasks) >= MAX_SUBTASKS:
                    log.warning(f"[extend-subtasks] hit cap {MAX_SUBTASKS}, "
                                f"rejecting further additions")
                    break
                if not isinstance(item, dict):
                    continue
                aspect = str(item.get("aspect", "") or "").strip()
                if not aspect:
                    continue
                mods = [m for m in (item.get("modalities") or [])
                        if str(m) in VALID_MODS]
                if not mods:
                    continue
                # New uniqueness: (aspect, modality) tuple. Reject duplicate
                # combo but allow same aspect with different modality (needed
                # for cross-modal verification subtasks).
                new_key = (aspect, tuple(sorted(mods)))
                if new_key in existing_keys:
                    log.warning(f"[extend-subtasks] skip duplicate "
                                f"(aspect, modalities)={new_key}")
                    continue
                sid = str(item.get("id") or
                          f"ext{len(state.subtasks or [])+i+1}")
                desc = str(item.get("description", ""))[:300]
                imp = max(0.0, min(1.0, float(item.get("importance", 0.5))))
                state.subtasks.append(
                    Subtask(id=sid, description=desc, aspect=aspect,
                            importance=imp, modalities=mods))
                existing_keys.add(new_key)
                added += 1
            log.info(f"[extend-subtasks] +{added} added "
                     f"(total now {len(state.subtasks)})")
        elif name == "REVISE_SUBTASK":
            # Mutate one subtask in-place. No-op if id not found.
            # Every revise is recorded to state.revise_trace for ablation
            # (can we disable REVISE and match baseline quality?).
            sid = args["id"]
            target = next((s for s in (state.subtasks or [])
                           if getattr(s, "id", None) == sid), None)
            if target is None:
                log.warning(f"[revise-subtask] id {sid!r} not found")
                return
            VALID_MODS = {"doc_text", "doc_visual",
                          "video_text", "video_text".replace("text","visual")}
            # Canonical version above has a typo risk; use literal set:
            VALID_MODS = {"doc_text", "doc_visual",
                          "video_text", "video_visual"}
            before = {"modalities": list(target.modalities),
                      "importance": target.importance,
                      "description": target.description}
            changes = []
            if "modalities" in args:
                mods = [m for m in args["modalities"] if str(m) in VALID_MODS]
                if mods:
                    target.modalities = mods
                    changes.append(f"modalities={mods}")
            if "importance" in args:
                target.importance = max(0.0, min(1.0, float(args["importance"])))
                changes.append(f"importance={target.importance:.2f}")
            if "description" in args:
                target.description = str(args["description"])[:300]
                changes.append("description=updated")
            log.info(f"[revise-subtask] {sid} {','.join(changes) or '(no-op)'}")
            if changes:
                state.revise_trace.append({
                    "turn": state.step_count,
                    "subtask_id": sid,
                    "before": before,
                    "after": {"modalities": list(target.modalities),
                              "importance": target.importance,
                              "description": target.description},
                    "changes": changes,
                })
        elif name == "ABORT_AND_RESPAWN":
            # Atomic kill+spawn. Inherits aspect by default, allows overrides.
            # Guardrail: each agent_id can only be ABORTed once per run, to
            # prevent DA from looping abort/respawn on the same problem.
            old_aid = args["agent_id"]
            if old_aid not in self.agents:
                log.warning(f"[abort-respawn] {old_aid} unknown")
                return
            if old_aid in state.aborted_agents:
                log.warning(f"[abort-respawn] {old_aid} already aborted once; "
                            f"rejecting second abort (DA must accept current "
                            f"result or try a different aspect)")
                return
            old_rec = self.agents[old_aid]
            # Mark aborted BEFORE spawning, so the new agent id isn't in the set
            state.aborted_agents.add(old_aid)
            # KILL the old agent
            old_rec.status = KILLED
            pool.set_agent_status(old_aid, "killed")
            log.info(f"[abort-respawn] killed {old_aid} "
                     f"(reason: {args.get('reason', 'n/a')})")
            # SPAWN a fresh one with overrides (or carry-over fields)
            spawn_args = {
                "agent_type": old_rec.agent_type,
                "modality":   args.get("new_modality") or old_rec.modality,
                "goal":       args.get("new_goal") or old_rec.goal,
                "aspect":     (args.get("new_aspect")
                              or getattr(old_rec, "aspect", None)),
            }
            self._spawn(spawn_args, state, pool)
        elif name == "RESOLVE_CONFLICT":
            # DA declares how a cross-modal aspect conflict is settled.
            # Three resolution kinds:
            #   trust_one     — adopt trust_agent_id's finding; others ignored
            #                   for this aspect at synthesis time
            #   unresolvable  — conflict is fundamental; answer must mention it
            #   complementary — findings aren't really in conflict, they cover
            #                   different facets; keep all at synthesis
            aspect = args["aspect"]
            resolution = args["resolution"]
            if resolution not in {"trust_one", "unresolvable", "complementary"}:
                log.warning(f"[resolve-conflict] invalid resolution "
                            f"{resolution!r}; ignoring")
                return
            trust_aid = args.get("trust_agent_id")
            if resolution == "trust_one" and not trust_aid:
                log.warning(f"[resolve-conflict] trust_one requires "
                            f"trust_agent_id; ignoring")
                return
            if trust_aid and trust_aid not in self.agents:
                log.warning(f"[resolve-conflict] unknown trust_agent_id "
                            f"{trust_aid!r}; ignoring")
                return
            entry = {
                "aspect": aspect,
                "resolution": resolution,
                "trust_agent_id": trust_aid,
                "reason": args.get("reason", ""),
                "turn": state.step_count,
            }
            state.conflict_resolutions[aspect] = entry
            log.info(f"[resolve-conflict] aspect={aspect} "
                     f"resolution={resolution} trust={trust_aid}")
        elif name == "STOP_TASK":
            state.finalize(answer=args.get("answer", "insufficient evidence"),
                           confidence=args.get("confidence",
                                               pool.max_confidence()),
                           reason=args["reason"])
            log.info(f"[stop] {args['reason']}")
        else:
            # Defensive: validate_command should have caught this, but log
            # anyway to surface any schema/dispatcher drift during dev.
            log.warning(f"[unhandled-command] {name}")

    def _spawn(self, args, state=None, pool=None):
        # Round 4 — saturation hard rule. When info-gain has stalled over
        # the rolling window, reject new sub-agent spawns: the system will
        # not benefit from more retrieval. DA must STOP or, rarely, INSPECT.
        if state is not None:
            tr = getattr(state, "info_gain_tracker", None)
            if tr is not None and tr.is_saturated():
                log.warning(f"[spawn-rejected] info-gain saturated; "
                            f"new agents denied. DA must STOP_TASK or "
                            f"RESOLVE_CONFLICT on existing evidence.")
                return
        # Plan-consistency check (P2 round-1, guardrail #9):
        # When state.subtasks is populated (decomposition succeeded), require
        # SPAWN to reference an existing (aspect, modality) pair from the plan.
        # This prevents DA from drifting off its own decomposition mid-run.
        # If state.subtasks is empty (decompose disabled or failed), skip
        # the check — backward-compat for tests and scripted providers.
        if state is not None and getattr(state, "subtasks", None):
            subtasks = state.subtasks
            aspect = args.get("aspect")
            if not aspect:
                log.warning(f"[spawn-rejected] aspect missing but subtasks "
                            f"plan exists; DA must tag spawns with a plan "
                            f"aspect. Rejecting.")
                return
            modality = args["modality"]
            # Accept if ANY subtask has this (aspect, modality) pair. We
            # don't require exact subtask_id match so DA can spawn multiple
            # agents against the same subtask (different goals/queries).
            matched = any(
                getattr(s, "aspect", None) == aspect and
                modality in (getattr(s, "modalities", []) or [])
                for s in subtasks)
            if not matched:
                planned_pairs = sorted({
                    (s.aspect, m) for s in subtasks for m in s.modalities})
                log.warning(f"[spawn-rejected] (aspect={aspect!r}, "
                            f"modality={modality!r}) not in plan. "
                            f"Planned: {planned_pairs}")
                return

        aid = new_agent_id()
        # Sub-agents skip the internal intent stage — Orchestrator registers
        # them directly for summary (Tier-1). Sub-agent workflow is now
        # 2 tool calls (retrieval, write_evidence(summary)). This removes one
        # decision point that 7B models frequently botched. Intent is only
        # used internally for trace/debug, not shown to Decision Agent.
        rec = AgentRecord(
            agent_id=aid, agent_type=args["agent_type"],
            modality=args["modality"], goal=args["goal"],
            aspect=args.get("aspect"),
            stage="summary",                 # skip intent stage entirely
            autonomy_level="autonomous",
            target_stage="summary",
        )
        self.agents[aid] = rec
        inst = self.factory.create(role=args["agent_type"],
                                   task=args["goal"],
                                   modality=rec.modality)
        inst.id = aid
        self._instances[aid] = inst
        if pool is not None:
            pool.register_agent(aid, rec.modality, rec.goal,
                                aspect=args.get("aspect"))
            try:
                pool.write(aid, "intent", {
                    "modality": rec.modality,
                    "data_source": f"{rec.modality}_index",
                    "planned_k": args.get("top_k", 10),
                })
            except Exception as e:
                log.warning(f"[auto-intent-fail] {aid}: {e}")
        self.queue.push(Task(new_task_id(), aid, {"kind": "step"}))
        if state is not None:
            state.record_spawn(aid)
        log.info(f"[spawn] {aid} modality={rec.modality} "
                 f"goal={rec.goal[:60]!r}")

    def _require(self, aid) -> AgentRecord:
        if aid not in self.agents:
            raise KeyError(f"unknown agent {aid}")
        return self.agents[aid]

    # ---------- PRE-RUN DECOMPOSITION ----------
    def _pre_run_decompose(self, query: str, state) -> None:
        """Ask the DA's provider to decompose the query into subtasks via
        the DECOMPOSE skill. Populates state.subtasks with validated,
        embedded, and deduped Subtask objects. Best-effort: on any failure
        state.subtasks stays empty and VoI still works via other hard rules.
        """
        from prompts.skills import decompose_skill
        provider = getattr(self.decision_agent, "_provider", None)
        arr = decompose_skill.run(query, provider)
        if not arr:
            return
        if getattr(state, "budget", None) is not None:
            # Decompose sees the query only; system prompt is ~2.5K tokens
            approx = 2500 + len(query) // 3
            state.budget.tier_breakdown["skill_decompose"] = (
                state.budget.tier_breakdown.get("skill_decompose", 0) + approx)

        from orchestrator.runtime import Subtask
        VALID_MODS = {"doc_text", "doc_visual", "video_text", "video_visual"}
        out: list = []
        seen_keys: set = set()
        for i, item in enumerate(arr[:10]):
            if len(out) >= 5:
                break
            if not isinstance(item, dict):
                continue
            try:
                sid = str(item.get("id") or f"s{i+1}")
                desc = str(item.get("description", ""))[:300]
                aspect = str(item.get("aspect", "") or sid).strip()
                importance = float(item.get("importance", 0.5))
                importance = max(0.0, min(1.0, importance))
                mods_raw = item.get("modalities") or []
                if not isinstance(mods_raw, list):
                    mods_raw = []
                mods = [str(m) for m in mods_raw if str(m) in VALID_MODS]
                if not mods:
                    log.warning(f"[decompose] subtask {sid} has no valid "
                                f"modalities, skipping")
                    continue
                # Enforce "1 subtask = 1 modality". Split multi-modality
                # entries into per-modality siblings sharing the aspect.
                for mod in mods:
                    key = (aspect, mod)
                    if key in seen_keys:
                        log.warning(f"[decompose] duplicate (aspect,modality)="
                                    f"{key}, skipping")
                        continue
                    seen_keys.add(key)
                    split_id = sid if len(mods) == 1 else f"{sid}_{mod}"
                    out.append(Subtask(id=split_id, description=desc,
                                       aspect=aspect, importance=importance,
                                       modalities=[mod]))
                    if len(out) >= 5:
                        break
            except Exception:
                continue

        # Embed + dedup near-duplicate subtasks (purpose A).
        try:
            from utils.subtask_embedder import (embed_subtasks,
                                                dedupe_subtasks,
                                                MockCLIPTextEmbedder)
            embedder = getattr(self, "subtask_embedder", None) or \
                       MockCLIPTextEmbedder()
            embed_subtasks(out, embedder)
            pre_count = len(out)
            out = dedupe_subtasks(out, threshold=0.92)
            if len(out) < pre_count:
                log.info(f"[decompose] deduped {pre_count - len(out)} "
                         f"near-duplicate subtask(s)")
        except Exception as e:
            log.warning(f"[decompose-embed-fail] {e}")

        if out and not any(s.importance >= 0.7 for s in out):
            log.warning("[decompose] no subtask with importance>=0.7 — "
                        "decomposition may be miscalibrated")
        state.subtasks = out
        log.info(f"[decompose] produced {len(out)} subtask(s)")
        for s in out:
            log.info(f"[decompose]   {s.id} aspect={s.aspect} "
                     f"imp={s.importance:.2f} mods={s.modalities}: "
                     f"{s.description[:60]!r}")

    # ---------- VoI HELPERS ----------
    def _find_subtask_for(self, rec, state):
        """Match an AgentRecord to a Subtask by aspect (the common key)."""
        subtasks = getattr(state, "subtasks", None) or []
        aspect = getattr(rec, "aspect", None)
        if aspect:
            for s in subtasks:
                if getattr(s, "aspect", None) == aspect:
                    return s
        # Fallback: match by goal substring — best-effort, may miss.
        for s in subtasks:
            if s.description and rec.goal and s.description[:40] in rec.goal:
                return s
        return None

    def _voi_retry_count(self, agent_id: str, stage: str, state) -> int:
        """Count prior DENY decisions for the same (agent_id, stage). Used to
        implement option-A soft-block-with-retry: the 2nd request after a
        denial is auto-approved."""
        decisions = getattr(state, "voi_decisions", None) or []
        return sum(1 for d in decisions
                   if d.get("agent_id") == agent_id
                   and d.get("stage") == stage
                   and not d.get("allow", True))

    # ---------- PARALLEL STEP EXECUTION ----------
    def _run_queued_steps(self, state, pool):
        tasks = [t for t in self.queue.drain()
                 if self.agents.get(t.agent_id)
                 and self.agents[t.agent_id].status not in
                 (KILLED, DONE, FAILED)]
        if not tasks: return

        for t in tasks: self.agents[t.agent_id].status = RUNNING

        def runner(task):
            rec = self.agents[task.agent_id]
            if rec.autonomy_level == "autonomous":
                return lambda: self._run_autonomous(task, state, pool)
            return lambda: self._run_one_step(task, state, pool)

        results = self.workers.run_many([runner(t) for t in tasks])

        # Autonomous agents handle their own stage/status transitions;
        # supervised agents get their transition applied here.
        for ok, res in results:
            if not ok:
                log.error(f"[step-fail] {res}")
                continue
            # Result is either None (autonomous, already done) or
            # (task, action, rec) from a single supervised step.
            if res is None:
                continue
            task, action, rec = res
            try:
                self._execute_tool(action, state, pool, caller_id=rec.agent_id)
                if rec.stage != "full":
                    rec.stage = STAGES[STAGES.index(rec.stage) + 1]
                rec.status = PENDING
            except Exception as e:
                rec.retries_left -= 1
                log.error(f"[tool-fail] {rec.agent_id}: {e} "
                          f"retries_left={rec.retries_left}")
                rec.status = FAILED if rec.retries_left <= 0 else PENDING

    def _run_one_step(self, task, state, pool):
        """Supervised mode: run exactly ONE sub-agent step, return the action
        for the outer loop to dispatch + transition."""
        rec = self.agents[task.agent_id]
        inst = self._instances[task.agent_id]
        inst.task = rec.goal
        inst.modality = rec.modality
        inst._stage_idx = STAGES.index(rec.stage)
        inst.recent_actions = list(rec.recent_actions[-5:])  # last 5
        action = inst.step(state)
        return task, action, rec

    def _run_autonomous(self, task, state, pool):
        """Autonomous mode with STRICT 2-CALL LIFECYCLE.

        Sub-agents have exactly one valid sequence:
          Step 1: retrieval(...)
          Step 2: write_evidence(stage="summary", ...)

        The Orchestrator enforces this lifecycle at the dispatch boundary.
        Anything else is either rejected (and the sub-agent is re-prompted
        with a corrective feedback message) or, after 2 strikes, causes the
        sub-agent to be marked FAILED.

        Rationale: mid-tier LLMs (Qwen-Plus, DeepSeek-chat) loop when given
        multi-step tool workflows. Hard-coding the 2-step progression at
        the Orchestrator level bounds the cost at ~2 LLM calls per sub-agent.
        """
        rec = self.agents[task.agent_id]
        inst = self._instances[task.agent_id]
        # Hard cap: at most 4 inner LLM rounds (2 happy-path + 2 correction slots)
        MAX_INNER_ROUNDS = 4
        strikes = 0

        for round_idx in range(MAX_INNER_ROUNDS):
            if rec.status in (KILLED, FAILED):
                return None

            inst.task = rec.goal
            inst.modality = rec.modality
            inst._stage_idx = STAGES.index(rec.stage)
            inst.recent_actions = list(rec.recent_actions[-5:])
            inst.last_retrieval = list(rec.last_retrieval)
            inst.last_reranked = list(rec.last_reranked)

            try:
                action = inst.step(state)
            except Exception as e:
                rec.retries_left -= 1
                log.error(f"[auto-step-fail] {rec.agent_id}: {e}")
                if rec.retries_left <= 0:
                    rec.status = FAILED
                    return None
                continue

            tool_name = action["tool_name"]
            has_retrieved = any(t == "retrieval" for t in rec.recent_actions)
            has_written  = any(t == "write_evidence" for t in rec.recent_actions)

            # --- Enforce the 2-call contract ---
            valid = False
            if not has_retrieved:
                valid = (tool_name == "retrieval")
                if not valid:
                    state.push_feedback(rec.agent_id,
                        "You MUST call `retrieval` first. Any other tool "
                        "call is rejected. Format: retrieval(query, "
                        f"modality={rec.modality!r}, top_k=10).")
            elif not has_written:
                valid = (tool_name == "write_evidence")
                if not valid:
                    state.push_feedback(rec.agent_id,
                        "Retrieval done. You MUST now call "
                        "`write_evidence(stage=\"summary\", payload={...})` "
                        "citing the retrieved ids. Any other tool is rejected.")
            else:
                # Both required calls have happened. If authorized for sketch
                # or Tier-3, allow one more write_evidence. Otherwise terminate.
                if tool_name == "write_evidence" and (
                        pool.is_authorized_for_full(rec.agent_id) or
                        pool.is_authorized_for_sketch(rec.agent_id)):
                    valid = True
                else:
                    # Happy exit — sub-agent's 2-call life is complete.
                    rec.status = DONE
                    return None

            if not valid:
                strikes += 1
                log.warning(f"[guard] {rec.agent_id} wrong-tool strike "
                            f"{strikes}/2 (got {tool_name})")
                if strikes >= 2:
                    log.error(f"[guard] {rec.agent_id} FAILED: refused to "
                              f"follow 2-call contract after 2 strikes")
                    rec.status = FAILED
                    return None
                continue  # re-prompt; don't execute the invalid call

            # --- Execute the valid tool call ---
            try:
                tool_result = self._execute_tool(
                    action, state, pool, caller_id=rec.agent_id)
            except Exception as e:
                rec.retries_left -= 1
                log.error(f"[auto-tool-fail] {rec.agent_id}: {e}")
                if rec.retries_left <= 0:
                    rec.status = FAILED
                    return None
                continue

            # Cache retrieval result + register citation-valid ids WITH meta.
            # Using note_retrieved_candidates lets the pool enrich downstream
            # citations (from sub-agent) with asset_type / source / page etc.
            if tool_name == "retrieval" and isinstance(tool_result, list):
                rec.last_retrieval = tool_result
                if tool_result:
                    pool.note_retrieved_candidates(rec.agent_id, tool_result)

            rec.recent_actions.append(tool_name)
            if len(rec.recent_actions) > 8:
                rec.recent_actions = rec.recent_actions[-8:]

            # Happy exit after summary write
            if tool_name == "write_evidence":
                written_stage = action["arguments"].get("stage", "summary")
                if written_stage in ("summary", "full"):
                    rec.status = DONE
                    return None

        # Exceeded inner rounds — likely stuck. Mark FAILED.
        log.warning(f"[guard] {rec.agent_id} exhausted MAX_INNER_ROUNDS")
        rec.status = FAILED
        return None

    # ---------- TOOL EXECUTION ----------
    def _execute_tool(self, action, state, pool, caller_id):
        spec = ToolRegistry.get(action["tool_name"])
        args = dict(action["arguments"])
        if args.get("agent_id") == "__self__":
            args["agent_id"] = caller_id
        # Expose the Decision Agent's provider to tools that need an LLM
        # (e.g. decompose_query). Sub-agent tools generally don't use this.
        # Round 3: also expose the caller's assigned modality so tools like
        # retrieval_text/retrieval_visual can route to the right index
        # without requiring the sub-agent to pass modality as an argument.
        caller_rec = self.agents.get(caller_id)
        agent_modality = getattr(caller_rec, "modality", "") if caller_rec else ""
        ctx = {"state": state, "pool": pool,
               "store": self.store, "factory": self.factory,
               "agent_modality": agent_modality,
               "provider": getattr(self.decision_agent, "_provider", None)}

        # === Cross-modal curated evidence injection ===
        # If sub-agent is writing curated stage and has pre-selected candidates,
        # inject them into the payload to enable cross-modal evidence alignment.
        if action["tool_name"] == "write_evidence":
            stage = args.get("stage", "")
            if stage == "curated" and caller_rec:
                curated_candidates = getattr(caller_rec, "_curated_candidates", None)
                if curated_candidates:
                    from utils.cross_modal_selector import CrossModalSelector
                    formatted = CrossModalSelector.format_for_subagent(
                        curated_candidates
                    )
                    # Get aspect_agreements for cross-modal analysis
                    aspect_agreements = pool.aspect_agreements()

                    # Inject formatted curated into payload (merge with sub-agent's payload)
                    original_payload = args.get("payload", {})
                    if isinstance(original_payload, dict):
                        # Merge: sub-agent's payload takes precedence for structure,
                        # but we inject key_candidates from CrossModalSelector
                        merged_payload = {**formatted, **original_payload}
                        args["payload"] = merged_payload

                        # Inject aspect_agreements into context for curated write
                        ctx["_aspect_agreements"] = aspect_agreements

                        conflict_aspects = CrossModalSelector.identify_conflict_aspects(
                            aspect_agreements
                        )
                        log.info(f"[exec] write_evidence caller={caller_id} "
                                 f"stage=curated injected_n={len(curated_candidates)} "
                                 f"conflict_aspects={conflict_aspects}")

        result = spec.handler(**args, **ctx)
        state.record({"tool": action["tool_name"], "caller": caller_id,
                      "args": args})
        log.info(f"[exec] {action['tool_name']} caller={caller_id}")
        return result
