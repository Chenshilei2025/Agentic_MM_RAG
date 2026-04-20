"""Agents are PURE POLICY.

- DecisionAgent emits COMMANDS ({"command": ..., "arguments": ...}).
- SubAgent (Seeker+Inspector) emits TOOL CALLS ({"tool_name": ..., "arguments": ...}).
- Neither executes anything. Orchestrator dispatches both.

Sub-agents are STATELESS: stage/modality/goal are injected by the Orchestrator
before each step. Sub-agents CANNOT spawn other agents — the prompt charter
forbids it AND the tool whitelist excludes spawn_agent/kill_agent.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import uuid, json
from cli.schemas.action import parse as parse_json, _extract_first_json_object
from cli.validator import validate as validate_action
from cli.schemas.commands import validate_command
from prompts.prompt_builder import PromptBuilder


class BaseAgent(ABC):
    def __init__(self, provider):
        self.id = str(uuid.uuid4())
        self._provider = provider

    def _complete(self, prompt: Dict[str, Any]) -> Dict[str, Any]:
        raw = self._provider.complete(
            messages=prompt["messages"], system=prompt["system"])
        obj_str = _extract_first_json_object(raw)
        return json.loads(obj_str)

    @abstractmethod
    def step(self, state) -> Dict[str, Any]: ...


class DecisionAgent(BaseAgent):
    """Emits structured COMMANDS only. Does not call tools directly."""

    def step(self, state) -> Dict[str, Any]:
        pool = state.pool
        board = pool.status_board()
        recent = state.drain_recent_spawns()
        archived = pool.archived_agents()
        budget = {
            "steps_used": state.step_count,
            "steps_max":  getattr(state, "max_steps", None),
            "tokens_used": (getattr(self._provider, "total_input_tokens", 0)
                          + getattr(self._provider, "total_output_tokens", 0)),
        }
        # Consume pending INSPECT blocks once: move them to an injected list
        # that PromptBuilder embeds into the multimodal user message, then clear.
        pending_inspect = list(getattr(state, "pending_inspect", []))
        state.pending_inspect = []
        # Same one-shot semantics for tier-3 archive content the DA requested
        # via READ_ARCHIVE on a previous turn.
        pending_archive = list(getattr(state, "pending_archive", []))
        state.pending_archive = []
        # Collect any available Tier-2.5 sketches (shown to DA as text blocks)
        sketches = {}
        for row in board:
            if row.get("sketch_available"):
                s = pool.get_sketch(row["agent_id"])
                if s: sketches[row["agent_id"]] = s
        # Round 2 — collect Tier-2 curated blocks (new unified disclosure)
        curated_blocks = {}
        for row in board:
            if row.get("curated_available"):
                c = pool.get_curated(row["agent_id"])
                if c: curated_blocks[row["agent_id"]] = c
        # Round 2 — one-shot consume the latest REFLECT verdict (if any)
        reflect_verdict = getattr(state, "pending_reflect_verdict", None)
        state.pending_reflect_verdict = None
        # P2 — pull aspect-level agreement signals
        aspect_agreements = pool.aspect_agreements()
        prompt = PromptBuilder.decision(
            query=state.query,
            status_board=board,
            coverage=pool.coverage(),
            max_conf=pool.max_confidence(),
            step=state.step_count,
            recent_spawns=recent,
            archived_agents=archived,
            budget=budget,
            pending_inspect=pending_inspect,
            sketches=sketches,
            curated_blocks=curated_blocks,
            reflect_verdict=reflect_verdict,
            pending_archive=pending_archive,
            subtasks=getattr(state, "subtasks", None),
            budget_state=getattr(state, "budget", None),
            voi_decisions=getattr(state, "voi_decisions", None),
            aspect_agreements=aspect_agreements,
            conflict_resolutions=dict(getattr(state, "conflict_resolutions", {})),
            info_gain_snapshot=(state.info_gain_tracker.snapshot()
                                if hasattr(state, "info_gain_tracker")
                                else None),
        )
        raw = self._complete(prompt)
        return validate_command(raw)


class SubAgent(BaseAgent):
    """Seeker+Inspector fused. STATELESS — stage & modality injected per step."""
    def __init__(self, provider, allowed_tools: List[str], role: str, task: str):
        super().__init__(provider)
        self.role = role
        self.task = task
        self.modality = "text"          # injected by Orchestrator before step
        self._stage_idx = 0             # injected by Orchestrator before step
        self._allowed = allowed_tools

    @property
    def stage(self) -> str:
        return ("intent", "summary", "full")[min(self._stage_idx, 2)]

    def step(self, state) -> Dict[str, Any]:
        fb = state.drain_feedback(self.id)
        recent = getattr(self, "recent_actions", [])
        last_retrieval = getattr(self, "last_retrieval", [])
        last_reranked  = getattr(self, "last_reranked", [])
        # Prefer reranked results if available; fall back to raw retrieval.
        recent_results = last_reranked or last_retrieval
        prompt = PromptBuilder.subagent(
            role=self.role, task=self.task, stage=self.stage,
            feedback=fb, allowed_tools=self._allowed,
            step=state.step_count, recent_actions=recent,
            recent_results=recent_results,
            modality=getattr(self, "modality", None),
        )
        raw_str = self._provider.complete(messages=prompt["messages"],
                                          system=prompt["system"])
        action = validate_action(parse_json(raw_str))
        if action["tool_name"] not in self._allowed:
            raise PermissionError(
                f"subagent {self.id} not permitted tool {action['tool_name']}")
        if action["tool_name"] in ("spawn_agent", "stop_agent"):
            raise PermissionError(
                "sub-agents cannot modify system topology")
        return action
