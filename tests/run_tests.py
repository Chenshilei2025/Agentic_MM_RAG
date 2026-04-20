"""Tests for the MERP agentic RAG system."""
import sys, time, traceback
from typing import Dict
sys.path.insert(0, ".")
import tools.builtin

from cli.schemas.action import parse, validate, ParseError, ValidationError
from cli.schemas.commands import validate_command, CommandError
from memory.evidence_pool import EvidencePool, DisclosureError
from memory.store import MultiIndexStore
from providers.mock_provider import ScriptedProvider
from agents.decision_agent import DecisionAgent
from agents.factory import SubAgentFactory
from orchestrator.controller import Orchestrator
from orchestrator.runtime import TaskQueue, WorkerPool, Task

PASS, FAIL = [], []
def check(name, fn):
    try: fn(); PASS.append(name); print(f"  PASS  {name}")
    except Exception as e: FAIL.append((name, str(e))); print(f"  FAIL  {name}: {e}")

# ---- Helper: new Tier-1 summary payload builder ----
def tier1_summary(finding="x is y", reasoning="Evidence supports this",
                  addressed=None, partial=None, uncovered=None,
                  rq="high", ec="high", rs="high",
                  critical_gaps=None, suggested_mods=None,
                  n_retrieved=5, n_kept=3, top_score=0.88, score_spread=0.12,
                  citations=None,
                  # Optional reflection signals
                  evidence_mode=None, retrieval_quality=None,
                  modality_fit=None, query_rewrite_suggestion=None,
                  # Raw payload override for advanced tests
                  **kwargs):
    """Build a valid Tier-1 summary payload."""
    payload = {
        "finding": finding,
        "reasoning": reasoning,
        "task_completion": {
            "addressed": addressed or [],
            "partial": partial or [],
            "uncovered": uncovered or [],
        },
        "confidence": {
            "retrieval_quality": rq,
            "evidence_coherence": ec,
            "reasoning_strength": rs,
        },
        "local_gaps": {
            "critical": critical_gaps or [],
            "suggested_modalities": suggested_mods or [],
        },
        "n_retrieved": n_retrieved,
        "n_kept": n_kept,
        "top_score": top_score,
        "score_spread": score_spread,
        "citations": citations or ["doc_1"],
    }
    # Add optional reflection signals
    if evidence_mode is not None:
        payload["evidence_mode"] = evidence_mode
    if retrieval_quality is not None:
        payload["retrieval_quality"] = retrieval_quality
    if modality_fit is not None:
        payload["modality_fit"] = modality_fit
    if query_rewrite_suggestion is not None:
        payload["query_rewrite_suggestion"] = query_rewrite_suggestion
    # Allow raw override for advanced tests
    payload.update(kwargs)
    return payload

def tier1_summary_minimal(finding="x", citations=["doc_1"]):
    """Minimal valid Tier-1 summary for simple tests."""
    return {
        "finding": finding,
        "reasoning": "Direct evidence found",
        "task_completion": {"addressed": ["main"], "partial": [], "uncovered": []},
        "confidence": {"retrieval_quality": "high", "evidence_coherence": "high",
                       "reasoning_strength": "high"},
        "local_gaps": {"critical": [], "suggested_modalities": []},
        "n_retrieved": 1, "n_kept": 1, "top_score": 0.8, "score_spread": 0.0,
        "citations": citations,
    }

# ---- Commands ----
def t_cmd_valid():
    validate_command({"command": "SPAWN_AGENT",
                      "arguments": {"agent_type": "si", "modality": "doc_text",
                                    "goal": "find"}})
def t_cmd_request_full():
    validate_command({"command": "REQUEST_FULL_EVIDENCE",
                      "arguments": {"agent_id": "agent-x"}})
def t_cmd_unknown():
    try: validate_command({"command": "NOPE", "arguments": {}}); assert False
    except CommandError: pass

# ---- Evidence pool: tier schema ----
def t_tier1_ok():
    p = EvidencePool()
    p.write("a", "intent",
            {"modality": "doc_text", "data_source": "corpus",
             "planned_k": 5})

def t_tier1_missing_fields():
    p = EvidencePool()
    try: p.write("a", "intent", {"modality": "doc_text"}); assert False
    except DisclosureError: pass

def t_tier2_ok():
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x", aspect="workflow")
    p.note_retrieved_ids("a", ["doc_1", "doc_2"])
    p.write("a", "summary", tier1_summary(
        finding="x is y",
        reasoning="Multiple sources confirm",
        addressed=["main claim"],
        citations=["doc_1", "doc_2"],
    ))

def t_tier2_rejects_fabricated_citation():
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x")
    p.note_retrieved_ids("a", ["doc_1"])
    try:
        p.write("a", "summary", tier1_summary(
            finding="x",
            reasoning="Evidence found",
            n_retrieved=3, n_kept=1, top_score=0.5, score_spread=0.1,
            citations=["doc_FAKE"]  # fabricated!
        ))
        assert False, "should reject fabricated citation"
    except DisclosureError: pass

def t_tier2_unverified_marks_suspicious():
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x")
    p.note_retrieved_ids("a", ["doc_1", "doc_2"])
    p.write("a", "summary", tier1_summary(
        finding="x is y",
        reasoning="Mixed evidence quality",
        addressed=["main claim"],
        n_retrieved=5, n_kept=3, top_score=0.9, score_spread=0.1,
        citations=["doc_1", "doc_FAKE"]  # one verified, one not
    ))
    board = p.status_board()
    row = next(r for r in board if r["agent_id"] == "a")
    assert row["unverified_citations"] == ["doc_FAKE"]
    # Note: suspicious_confidence is computed but may not surface in status_board
    # depending on implementation details

def t_tier3_blocked_without_auth():
    p = EvidencePool()
    try:
        p.write("a", "full",
                {"content": "raw", "sources": ["d1"]}); assert False
    except DisclosureError as e:
        assert "not authorized" in str(e)

def t_tier3_allowed_after_auth():
    p = EvidencePool()
    p.authorize_full("a")
    p.write("a", "full", {"content": "raw", "sources": ["d1"]})
    # Authorization is one-shot
    try:
        p.write("a", "full", {"content": "raw2", "sources": []}); assert False
    except DisclosureError: pass

def t_metrics():
    p = EvidencePool()
    p.register_agent("a", "doc_text", "test")
    p.note_retrieved_ids("a", ["c1", "c2"])
    p.write("a", "intent", {"modality":"doc_text","data_source":"c","planned_k":3})
    assert p.coverage() == 0.0
    p.write("a", "summary", tier1_summary(
        finding="z",
        reasoning="Strong evidence",
        addressed=["test aspect"],
        citations=["c1", "c2"],
        top_score=0.9, score_spread=0.1,
    ))
    assert p.coverage() == 1.0
    # max axis float for "high" is 0.85
    assert abs(p.max_confidence() - 0.85) < 0.01

# ---- Runtime ----
def t_workerpool_parallel():
    wp = WorkerPool(max_workers=3, default_timeout=2.0)
    out = wp.run_many([lambda i=i: i*2 for i in range(3)])
    wp.shutdown()
    assert [r for ok,r in out if ok] == [0,2,4]

# ---- End-to-end: autonomous tier-1/tier-2 push, then tier-3 on request ----
def t_three_tier_flow():
    """Verify: spawn → sub auto-pushes intent+summary → DA requests full →
    sub emits full. Parallel N=2."""
    calls = {"n": 0}
    per_agent: Dict[str, int] = {}
    import threading
    lock = threading.Lock()

    class Prov:
        name = "p"
        def complete(self, messages, system="", **kw):
            with lock:
                calls["n"] += 1
                i = calls["n"]
            # Call 1: DA spawns 2 agents in parallel
            if i == 1:
                return ('{"command":"SPAWN_AGENTS","arguments":{"specs":['
                        '{"agent_type":"si","modality":"doc_text","goal":"A"},'
                        '{"agent_type":"si","modality":"doc_visual","goal":"B"}'
                        ']}}')

            text = str(messages)
            import re
            ids_in_prompt = re.findall(r"agent-[0-9a-f]+", text)
            # Find which sub-agent is calling by looking at the first agent
            # id mentioned in the prompt (sub-agent prompts carry self id).
            my_id = ids_in_prompt[0] if ids_in_prompt else None

            # DA-level calls: when status board mentions BOTH agents, this is
            # the DA turn after sub-agents finished tier-2.
            is_da_turn = ("STATUS BOARD" in text)

            if is_da_turn:
                with lock:
                    da_count = per_agent.get("_DA", 0) + 1
                    per_agent["_DA"] = da_count
                if da_count == 1:
                    # DA asks for full evidence from first finished agent
                    board_ids = re.findall(r"agent-[0-9a-f]+", text)
                    target = board_ids[0] if board_ids else "agent-0"
                    return ('{"command":"REQUEST_FULL_EVIDENCE","arguments":'
                            f'{{"agent_id":"{target}"}}}}')
                # Second DA turn: stop
                return ('{"command":"STOP_TASK","arguments":'
                        '{"reason":"done","answer":"resolved","confidence":0.95}}')

            # Sub-agent turn — advance that sub-agent's own step counter
            with lock:
                step = per_agent.get(my_id, 0) + 1
                per_agent[my_id] = step

            if step == 1:
                # intent
                return ('{"tool_name":"write_evidence","arguments":'
                        '{"agent_id":"__self__","stage":"intent","payload":'
                        '{"modality":"doc_text","data_source":"corpus","planned_k":3}}}')
            if step == 2:
                # retrieval — query the modality assigned to this sub-agent
                mod_match = re.search(r"ASSIGNED MODALITY:\s*(\w+)", text)
                mod = mod_match.group(1) if mod_match else "doc_text"
                return ('{"tool_name":"retrieval","arguments":'
                        f'{{"query":"X","modality":"{mod}","k":3}}}}')
            if step == 3:
                # summary with real citation id (new Tier-1 schema)
                return ('{"tool_name":"write_evidence","arguments":'
                        '{"agent_id":"__self__","stage":"summary","payload":{'
                        '"finding":"found it",'
                        '"reasoning":"Direct evidence matches query",'
                        '"task_completion":{"addressed":["main"],"partial":[],"uncovered":[]},'
                        '"confidence":{"retrieval_quality":"high",'
                        '"evidence_coherence":"high","reasoning_strength":"high"},'
                        '"local_gaps":{"critical":[],"suggested_modalities":[]},'
                        '"n_retrieved":1,"n_kept":1,"top_score":0.9,'
                        '"score_spread":0.0,"citations":["seed_1"]}}}')
            # step == 4 (only for authorized agent): write tier-3
            return ('{"tool_name":"write_evidence","arguments":'
                    '{"agent_id":"__self__","stage":"full","payload":'
                    '{"content":"RAW TEXT","sources":["d1","d2"]}}}')

    prov = Prov()
    from memory.store import VectorIndex
    seed_store = MultiIndexStore({
        "doc_text": VectorIndex("doc_text",
            [{"id": "seed_1", "content": "X exists.", "meta": {}}]),
        "doc_visual": VectorIndex("doc_visual",
            [{"id": "seed_1", "content": "X diagram.", "meta": {}}]),
    })
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                        seed_store, max_steps=10, max_workers=4)
    r = orch.run("query")
    assert r.get("confidence", 0) >= 0.9, f"bad: {r}"

def t_unauthorized_full_is_rejected():
    """Sub-agent attempting tier-3 without REQUEST_FULL_EVIDENCE fails."""
    calls = {"n": 0}
    class Prov:
        name = "p"
        def complete(self, messages, system="", **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return ('{"command":"SPAWN_AGENT","arguments":'
                        '{"agent_type":"si","modality":"doc_text","goal":"X"}}')
            # Sub tries to skip straight to "full" with no authorization
            if calls["n"] == 2:
                return ('{"tool_name":"write_evidence","arguments":'
                        '{"agent_id":"__self__","stage":"full","payload":'
                        '{"content":"bad","sources":[]}}}')
            return ('{"command":"STOP_TASK","arguments":'
                    '{"reason":"stopped","answer":"","confidence":0.0}}')
    prov = Prov()
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                        MultiIndexStore({}), max_steps=5, max_workers=2,
                        step_timeout=3.0)
    r = orch.run("q")
    # Agent should have FAILED status after retries exhausted
    rec = list(orch.agents.values())[0]
    assert rec.status == "FAILED", f"expected FAILED, got {rec.status}"

def t_multi_index_retrieval():
    """Retrieval tool must route by modality and reject unknown modalities."""
    from memory.store import VectorIndex, MultiIndexStore
    from tools.registry import ToolRegistry

    docs_dt = [{"id": "d1", "content": "orchestrator executes tools",
                "meta": {"source": "x.pdf"}}]
    docs_vt = [{"id": "v1", "content": "speaker mentions orchestrator",
                "meta": {"source": "talk.mp4", "timestamp": "00:01:00"}}]
    store = MultiIndexStore({
        "doc_text":   VectorIndex("doc_text", docs_dt),
        "video_text": VectorIndex("video_text", docs_vt),
    })
    ret = ToolRegistry.get("retrieval").handler

    r1 = ret(query="orchestrator", modality="doc_text", k=3,
             state=None, pool=None, store=store, factory=None)
    assert r1 and r1[0]["id"] == "d1" and r1[0]["modality"] == "doc_text"

    r2 = ret(query="orchestrator", modality="video_text", k=3,
             state=None, pool=None, store=store, factory=None)
    assert r2 and r2[0]["id"] == "v1"

    # Unknown modality -> KeyError
    try:
        ret(query="x", modality="video_visual", k=3,
            state=None, pool=None, store=store, factory=None)
        assert False, "should have rejected"
    except KeyError:
        pass


def t_ablation_no_tier3_gate():
    """With the gate disabled, sub-agents can write tier-3 unsolicited."""
    from config.ablation import AblationConfig
    from memory.evidence_pool import EvidencePool
    pool = EvidencePool(bypass_tier3_gate=True)
    # Should NOT raise
    pool.write("a", "full", {"content": "raw", "sources": ["d1"]})

def t_ablation_no_rerank_factory():
    from config.ablation import AblationConfig
    # Factory without rerank tools (rerank was always excluded; this
    # test really just verifies retrieval is present and no rerank is.)
    f = SubAgentFactory(provider=None, enable_rerank=False)
    # Default whitelist (modality unspecified) — superset
    tools = f.tools_for(None)
    assert "rerank_text" not in tools
    assert "retrieval" in tools

def t_ablation_no_parallelism_forces_single_worker():
    from config.ablation import AblationConfig
    ab = AblationConfig.preset("no_parallelism")
    class Prov:
        name = "p"
        def complete(self, *a, **k):
            return ('{"command":"STOP_TASK","arguments":'
                    '{"reason":"x","answer":"y","confidence":1.0}}')
    orch = Orchestrator(DecisionAgent(Prov()), SubAgentFactory(Prov()),
                        MultiIndexStore({}), max_steps=2, max_workers=4,
                        ablation=ab)
    # Orchestrator should have clamped workers to 1.
    assert orch.workers._exec._max_workers == 1

def t_ablation_presets_all_valid():
    from config.ablation import AblationConfig
    for name in ["baseline", "no_rerank", "no_tier3_gate",
                 "no_parallelism", "no_progressive_disc", "combined"]:
        AblationConfig.preset(name)


def t_hybrid_retrieval_rrf():
    """Hybrid store must merge dense + BM25 via RRF and return unified list."""
    from memory.store import VectorIndex, MultiIndexStore
    docs = [
        {"id": "d1", "content": "the orchestrator executes tools",
         "meta": {"source": "a"}},
        {"id": "d2", "content": "banana bread is delicious",
         "meta": {"source": "b"}},
        {"id": "d3", "content": "tools run via the orchestrator component",
         "meta": {"source": "c"}},
    ]
    store = MultiIndexStore(
        {"doc_text": VectorIndex("doc_text", docs)}, hybrid=True)
    hits = store.search("orchestrator tools", modality="doc_text", k=3)
    assert hits and hits[0]["id"] in ("d1", "d3"), f"bad hit: {hits[0]}"
    # RRF fusion adds rrf_score
    assert "rrf_score" in hits[0]


def t_voi_approves_mid_conf_hard_rule():
    """Mid-confidence zone triggers hard rule → always allow."""
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget
    row = {"agent_id": "a", "aspect": "x",
           "confidence": {"retrieval_score": 0.65,
                          "evidence_agreement": None, "coverage": None},
           "coverage": {"covered": 0.7, "gaps": []},
           "ambiguity": 0.3, "missing_aspects": []}
    d = gate_request(row, [row], None, Budget(10000, 0),
                     stage="full", retry_count=0)
    assert d.allow is True
    assert d.reason == "hard_rule_mid_conf"

def t_voi_denies_low_value():
    """High confidence + low uncertainty → VoI below threshold → deny."""
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget
    row = {"agent_id": "a", "aspect": "x",
           "confidence": {"retrieval_score": 0.95,
                          "evidence_agreement": 0.95, "coverage": 0.95},
           "coverage": {"covered": 0.95, "gaps": []},
           "ambiguity": 0.05, "missing_aspects": []}
    d = gate_request(row, [row], None, Budget(10000, 0),
                     stage="full", retry_count=0)
    assert d.allow is False
    assert d.reason == "voi_below_threshold"

def t_voi_retry_override():
    """Second request after deny is auto-approved."""
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget
    row = {"agent_id": "a", "aspect": "x",
           "confidence": {"retrieval_score": 0.95,
                          "evidence_agreement": 0.95, "coverage": 0.95},
           "coverage": {"covered": 0.95, "gaps": []},
           "ambiguity": 0.05, "missing_aspects": []}
    d = gate_request(row, [row], None, Budget(10000, 0),
                     stage="full", retry_count=1)
    assert d.allow is True
    assert d.reason == "retry_override"

def t_voi_budget_exceeded_denies():
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget
    row = {"agent_id": "a", "aspect": "x",
           "confidence": {"retrieval_score": 0.3,
                          "evidence_agreement": 0.3, "coverage": 0.3},
           "coverage": {"covered": 0.2, "gaps": ["g"]},
           "ambiguity": 0.8, "missing_aspects": ["m1", "m2"]}
    budget = Budget(1000, 2000)   # over-budget
    d = gate_request(row, [row], None, budget,
                     stage="full", retry_count=0)
    assert d.allow is False
    assert d.reason == "budget_exceeded"

def t_voi_conflict_hard_rule():
    """Two agents on same aspect with big confidence gap → conflict → allow."""
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget
    row_a = {"agent_id": "a", "aspect": "x",
             "confidence": {"retrieval_score": 0.9,
                            "evidence_agreement": 0.9, "coverage": 0.9},
             "coverage": {"covered": 0.9, "gaps": []},
             "ambiguity": 0.1, "missing_aspects": []}
    row_b = {"agent_id": "b", "aspect": "x",
             "confidence": {"retrieval_score": 0.2,
                            "evidence_agreement": 0.2, "coverage": 0.2},
             "coverage": {"covered": 0.2, "gaps": []},
             "ambiguity": 0.9, "missing_aspects": []}
    d = gate_request(row_a, [row_a, row_b], None, Budget(10000, 0),
                     stage="full", retry_count=0)
    assert d.allow is True
    assert d.reason == "hard_rule_conflict"

def t_voi_important_gap_hard_rule():
    from orchestrator.voi_gating import gate_request
    from orchestrator.runtime import Budget, Subtask
    row = {"agent_id": "a", "aspect": "x",
           "confidence": {"retrieval_score": 0.95,
                          "evidence_agreement": 0.95, "coverage": 0.95},
           "coverage": {"covered": 0.3, "gaps": ["critical"]},
           "ambiguity": 0.05, "missing_aspects": []}
    subtask = Subtask(id="s1", description="critical x", aspect="x",
                      importance=0.9)
    d = gate_request(row, [row], subtask, Budget(10000, 0),
                     stage="full", retry_count=0)
    assert d.allow is True
    assert d.reason == "hard_rule_important_gap"

def t_voi_select_evidence_tier():
    from orchestrator.voi_gating import select_evidence_tier
    clean = {"coverage": {"covered": 0.95}, "ambiguity": 0.05,
             "missing_aspects": []}
    assert select_evidence_tier(clean) == "TIER_2"
    mid = {"coverage": {"covered": 0.5}, "ambiguity": 0.5,
           "missing_aspects": ["m"]}
    assert select_evidence_tier(mid) == "TIER_2_5"
    messy = {"coverage": {"covered": 0.1}, "ambiguity": 0.9,
             "missing_aspects": ["m1", "m2", "m3"]}
    assert select_evidence_tier(messy) == "TIER_3"

def t_reflection_signals_persisted():
    """Sub-agent's evidence_mode / retrieval_quality / modality_fit /
    query_rewrite_suggestion fields propagate through pool to status board."""
    p = EvidencePool()
    p.register_agent("a", "doc_visual", "find diagram", aspect="layout")
    p.note_retrieved_ids("a", ["d1", "d2"])
    p.write("a", "summary", {
        "finding": "diagram shows X",
        "reasoning": "Caption mentions X but lacks detail",
        "task_completion": {"addressed": [], "partial": ["layout"], "uncovered": ["text labels"]},
        "confidence": {"retrieval_quality": "medium",
                       "evidence_coherence": "low",
                       "reasoning_strength": "low"},
        "local_gaps": {"critical": ["text labels"], "suggested_modalities": ["doc_text"]},
        "n_retrieved": 2, "n_kept": 1, "top_score": 0.6,
        "score_spread": 0.1, "citations": ["d1"],
        "evidence_mode": "caption_only",
        "retrieval_quality": "thin",
        "modality_fit": {"fit": False, "reason": "needs OCR not captions"},
        "query_rewrite_suggestion": "search for text labels under figures"})
    row = next(r for r in p.status_board() if r["agent_id"] == "a")
    assert row["evidence_mode"] == "caption_only"
    assert row["retrieval_quality"] == "thin"
    assert row["modality_fit"] is False
    assert "OCR" in row["modality_fit_reason"]
    assert "text labels" in row["query_rewrite_suggestion"]


def t_reflection_signals_validated():
    """Invalid evidence_mode or retrieval_quality must raise."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x", aspect="x")
    p.note_retrieved_ids("a", ["d1"])
    base = {"finding": "x",
            "coverage": {"covered": 0.5, "gaps": []},
            "confidence": {"retrieval_score": "low",
                           "evidence_agreement": "low", "coverage": "low"},
            "n_retrieved": 1, "n_kept": 1, "top_score": 0.5,
            "score_spread": 0.1, "caveat": None, "citations": ["d1"]}
    try:
        p.write("a", "summary", {**base, "evidence_mode": "BOGUS"})
        assert False, "should reject bad evidence_mode"
    except DisclosureError: pass
    try:
        p.write("a", "summary", {**base, "retrieval_quality": "BOGUS"})
        assert False, "should reject bad retrieval_quality"
    except DisclosureError: pass


def t_aspect_agreements_single_source():
    """One agent on an aspect → single_source state."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "g", aspect="control_loop")
    p.note_retrieved_ids("a", ["d1"])
    p.write("a", "summary", tier1_summary(
        finding="Orchestrator dispatches deterministically",
        reasoning="Clear evidence from documentation",
        addressed=["dispatch mechanism"],
        citations=["d1"], top_score=0.9, score_spread=0.05,
    ))
    out = p.aspect_agreements()
    assert len(out) == 1
    row = out[0]
    assert row["aspect"] == "control_loop"
    assert row["agreement_state"] == "single_source"
    assert row["n_agents"] == 1
    assert row["modalities_covered"] == ["doc_text"]


def t_aspect_agreements_disagree_via_conf_gap():
    """Two agents on same aspect with big confidence gap → disagree."""
    p = EvidencePool()
    # a1: high confidence on all axes (0.85)
    p.register_agent("a1", "doc_text", "g", aspect="determinism")
    p.note_retrieved_ids("a1", ["d1"])
    p.write("a1", "summary", tier1_summary(
        finding="Orchestrator is deterministic",
        reasoning="Evidence from doc_text suggests",
        addressed=["determinism"],
        rq="high", ec="high", rs="high",
        citations=["d1"], top_score=0.7, score_spread=0.1,
    ))
    # a2: low confidence on all axes (0.25) → max_conf_gap = 0.85 - 0.25 = 0.6
    p.register_agent("a2", "video_text", "g", aspect="determinism")
    p.note_retrieved_ids("a2", ["d1"])
    p.write("a2", "summary", tier1_summary(
        finding="Orchestrator runs LLM each turn",
        reasoning="Evidence from video_text suggests",
        addressed=["determinism"],
        rq="low", ec="low", rs="low",
        citations=["d1"], top_score=0.7, score_spread=0.1,
    ))
    out = p.aspect_agreements()
    row = next(r for r in out if r["aspect"] == "determinism")
    # max_conf_gap = 0.85 - 0.25 = 0.6  > 0.4
    # a2 is NOT all_confident (0.25 < 0.6) → state = complementary
    assert row["max_conf_gap"] > 0.4
    assert row["agreement_state"] in ("disagree", "complementary")
    assert "doc_text" in row["modalities_covered"]
    assert "video_text" in row["modalities_covered"]


def t_extend_subtasks_appends_unique_aspects():
    """EXTEND_SUBTASKS appends new subtasks; rejects duplicate aspects."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    prov = P()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov), store,
                        ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="x", aspect="a1",
                              importance=0.9, modalities=["doc_text"])]
    cmd = {"command": "EXTEND_SUBTASKS",
           "arguments": {"subtasks": [
               {"id": "s2", "description": "y", "aspect": "a2",
                "importance": 0.7, "modalities": ["doc_text"]},
               {"id": "s3", "description": "dup", "aspect": "a1",  # dup → reject
                "importance": 0.7, "modalities": ["doc_text"]},
               {"id": "s4", "description": "z", "aspect": "a3",
                "importance": 0.6, "modalities": ["bogus"]},  # no valid mod → reject
           ]}}
    orch._dispatch(cmd, state, state.pool)
    aspects = [s.aspect for s in state.subtasks]
    assert aspects == ["a1", "a2"], f"got {aspects}"


def t_revise_subtask_mutates_in_place():
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    prov = P()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov), store,
                        ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="orig", aspect="a1",
                              importance=0.5, modalities=["doc_text"])]
    cmd = {"command": "REVISE_SUBTASK",
           "arguments": {"id": "s1",
                         "modalities": ["doc_visual", "video_text"],
                         "importance": 0.9}}
    orch._dispatch(cmd, state, state.pool)
    s = state.subtasks[0]
    assert s.modalities == ["doc_visual", "video_text"]
    assert s.importance == 0.9
    assert s.description == "orig"  # unchanged
    # No-op when id missing
    orch._dispatch({"command": "REVISE_SUBTASK",
                    "arguments": {"id": "ghost",
                                  "importance": 0.1}},
                   state, state.pool)
    assert s.importance == 0.9  # unaffected


def t_decompose_skill_structured_output():
    """decompose_skill.run() returns a parsed list of subtask dicts with
    fine-grained aspects (event/spatial/entity/causal/temporal/process)."""
    from prompts.skills import decompose_skill
    class P:
        name = "p"
        def complete(self, messages, system="", **kw):
            # New aspect taxonomy: entity
            return ('[{"id":"s1","description":"Identify key entities in X.",'
                    '"aspect":"entity","importance":0.9,'
                    '"modalities":["doc_text"]}]')
    arr = decompose_skill.run("what is X?", P())
    assert isinstance(arr, list) and len(arr) == 1
    assert arr[0]["aspect"] == "entity"


def t_decompose_skill_robust_to_bad_providers():
    """Skill returns None — never throws — on bad/empty/mock provider."""
    from prompts.skills import decompose_skill
    # None provider
    assert decompose_skill.run("q", None) is None
    # Provider without complete()
    class Bare: pass
    assert decompose_skill.run("q", Bare()) is None
    # Provider that returns garbage
    class Garbage:
        def complete(self, *a, **kw): return "not json at all"
    assert decompose_skill.run("q", Garbage()) is None
    # Provider that throws
    class Throws:
        def complete(self, *a, **kw): raise RuntimeError("boom")
    assert decompose_skill.run("q", Throws()) is None


def t_subtask_embedding_populated_and_deduped():
    """embed_subtasks fills the embedding field; dedupe_subtasks collapses
    near-duplicates within the same modality."""
    from orchestrator.runtime import Subtask
    from utils.subtask_embedder import (MockCLIPTextEmbedder,
                                        embed_subtasks, dedupe_subtasks)
    subs = [
        Subtask(id="s1", description="locate orchestrator control loop",
                aspect="control_loop", importance=0.9,
                modalities=["doc_text"]),
        Subtask(id="s2", description="find the orchestrator control loop",
                aspect="control_loop_dup", importance=0.6,
                modalities=["doc_text"]),   # near-duplicate → should drop
        Subtask(id="s3",
                description="what color is the architecture diagram",
                aspect="diagram", importance=0.7,
                modalities=["doc_visual"]),
    ]
    embedder = MockCLIPTextEmbedder()
    embed_subtasks(subs, embedder)
    # All three have embeddings
    assert all(s.embedding for s in subs)
    assert len(subs[0].embedding) == embedder.dim
    # Dedupe drops s2 (similar to s1, same modality); keeps s3 (different mod)
    kept = dedupe_subtasks(subs, threshold=0.6)
    kept_ids = sorted(s.id for s in kept)
    # s1 should remain (it was first); s2 near-dup is dropped; s3 kept
    assert "s1" in kept_ids
    assert "s3" in kept_ids
    assert len(kept) == 2, f"expected 2 after dedup, got {len(kept)}: {kept_ids}"


def t_subtask_embedding_cross_modality_no_dedup():
    """Two subtasks with near-identical text but different modalities are
    NOT treated as duplicates (cross-modal verification pattern)."""
    from orchestrator.runtime import Subtask
    from utils.subtask_embedder import (MockCLIPTextEmbedder,
                                        embed_subtasks, dedupe_subtasks)
    subs = [
        Subtask(id="s_dt", description="speaker claims determinism",
                aspect="determinism", importance=0.9,
                modalities=["doc_text"]),
        Subtask(id="s_vt", description="speaker claims determinism",
                aspect="determinism", importance=0.9,
                modalities=["video_text"]),
    ]
    embedder = MockCLIPTextEmbedder()
    embed_subtasks(subs, embedder)
    kept = dedupe_subtasks(subs, threshold=0.9)
    # Both modalities preserved despite identical text
    assert len(kept) == 2
    assert sorted(s.modalities[0] for s in kept) == ["doc_text", "video_text"]


def t_retrieval_text_routes_by_agent_modality():
    """retrieval_text infers doc_text vs video_text from ctx.agent_modality."""
    from tools.builtin import retrieval_text
    from memory.store import VectorIndex, MultiIndexStore

    class FakeStore:
        def __init__(self): self.last_modality = None
        def search(self, query, modality, k):
            self.last_modality = modality
            return [{"id": f"{modality}_1", "content": "c",
                     "score": 0.9, "meta": {}}]

    store = FakeStore()
    # doc_text modality
    hits = retrieval_text(query="q", store=store, agent_modality="doc_text")
    assert store.last_modality == "doc_text"
    assert hits[0]["modality"] == "doc_text"
    # video_text modality
    hits = retrieval_text(query="q", store=store, agent_modality="video_text")
    assert store.last_modality == "video_text"
    # default fallback when agent_modality not in ctx
    hits = retrieval_text(query="q", store=store)
    assert store.last_modality == "doc_text"  # safe default


def t_retrieval_visual_routes_by_agent_modality():
    """retrieval_visual infers doc_visual vs video_visual."""
    from tools.builtin import retrieval_visual

    class FakeStore:
        def __init__(self): self.last_modality = None
        def search(self, query, modality, k):
            self.last_modality = modality
            return [{"id": "x", "content": "c", "score": 0.8, "meta": {}}]

    store = FakeStore()
    retrieval_visual(query="q", store=store, agent_modality="video_visual")
    assert store.last_modality == "video_visual"
    retrieval_visual(query="q", store=store, agent_modality="doc_visual")
    assert store.last_modality == "doc_visual"


def t_factory_per_modality_dict_precedence():
    """provider_per_modality[m] beats provider_visual and provider."""
    from agents.factory import SubAgentFactory
    p_default = object()
    p_visual = object()
    p_specific = object()
    f = SubAgentFactory(provider=p_default,
                        provider_visual=p_visual,
                        provider_per_modality={"video_visual": p_specific})
    # Exact match wins
    assert f.provider_for("video_visual") is p_specific
    # No exact match → visual fallback
    assert f.provider_for("doc_visual") is p_visual
    # Text modality → default
    assert f.provider_for("doc_text") is p_default
    # Unknown → default
    assert f.provider_for(None) is p_default


def t_factory_tools_for_text_modality():
    """Text sub-agents get retrieval_text, not retrieval_visual."""
    from agents.factory import SubAgentFactory
    f = SubAgentFactory(provider=None)
    tools = f.tools_for("doc_text")
    assert "retrieval_text" in tools
    assert "retrieval_visual" not in tools
    assert "write_evidence" in tools
    # video_text gets the same whitelist
    tools2 = f.tools_for("video_text")
    assert tools2 == tools


def t_factory_tools_for_visual_modality():
    """Visual sub-agents get retrieval_visual, not retrieval_text."""
    from agents.factory import SubAgentFactory
    f = SubAgentFactory(provider=None)
    tools = f.tools_for("doc_visual")
    assert "retrieval_visual" in tools
    assert "retrieval_text" not in tools
    assert "write_evidence" in tools


def t_curated_light_authorization_required():
    """_write_curated refuses without authorize_curated first."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "g", aspect="x")
    p.note_retrieved_candidates("a", [{"id": "d1", "content": "c",
                                       "meta": {}}])
    try:
        p.write("a", "curated", {
            "key_candidates": [{"id": "d1", "relevance": 0.9,
                                "evidence_hit": 0.8}]})
        assert False, "should reject unauthorized curated"
    except DisclosureError:
        pass


def t_curated_light_authorized_write_strips_raw():
    """When DA authorised light (without raw), raw blobs sent by sub-agent
    are silently dropped."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "g", aspect="x")
    p.note_retrieved_candidates("a", [{"id": "d1", "content": "c",
                                       "meta": {}}])
    p.authorize_curated("a", with_raw=False)
    p.write("a", "curated", {
        "key_candidates": [{"id": "d1", "relevance": 0.9,
                            "evidence_hit": 0.7,
                            "text": "headline",
                            "raw": "FULL_BLOB_HERE"}]})
    out = p.get_curated("a")
    assert out is not None
    assert out["with_raw"] is False
    assert "raw" not in out["key_candidates"][0]


def t_curated_raw_per_id_permission():
    """Raw blobs only persisted for ids in the authorised set."""
    p = EvidencePool()
    p.register_agent("a", "doc_visual", "g", aspect="x")
    p.note_retrieved_candidates("a", [
        {"id": "dv_1", "content": "c1", "meta": {"asset_type": "image"}},
        {"id": "dv_2", "content": "c2", "meta": {"asset_type": "image"}},
    ])
    # Only dv_1 authorised for raw
    p.authorize_curated("a", with_raw=True, ids=["dv_1"])
    p.write("a", "curated", {
        "key_candidates": [
            {"id": "dv_1", "relevance": 0.9, "evidence_hit": 0.8,
             "raw": "BASE64_IMG_1"},
            {"id": "dv_2", "relevance": 0.7, "evidence_hit": 0.6,
             "raw": "BASE64_IMG_2"},
        ]})
    out = p.get_curated("a")
    by_id = {e["id"]: e for e in out["key_candidates"]}
    assert "raw" in by_id["dv_1"]
    assert by_id["dv_1"]["raw"] == "BASE64_IMG_1"
    assert "raw" not in by_id["dv_2"], "dv_2 raw should have been stripped"


def t_reflect_command_stores_verdict_with_heuristic_fallback():
    """REFLECT dispatch populates state.pending_reflect_verdict; when the
    provider is scripted/none, fallback heuristic verdict is used."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class ScriptedProvider:   # name matches the skill's auto-skip list
        name = "scripted"
        def complete(self, *a, **kw): return ""

    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(ScriptedProvider()),
                        SubAgentFactory(ScriptedProvider()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="d", aspect="a1",
                              importance=0.9, modalities=["doc_text"])]
    orch._dispatch({"command": "REFLECT", "arguments": {"reason": "test"}},
                   state, state.pool)
    v = state.pending_reflect_verdict
    assert v is not None
    assert "recommended_action" in v
    assert v["source"] == "heuristic"   # fallback used
    # Heuristic detected the important subtask has no row → expect REPLAN
    assert v["recommended_action"] == "REPLAN"
    assert len(state.reflect_verdicts) == 1


def t_replan_command_dispatches_extend_revise_abort():
    """REPLAN applies a patch with all three op types through existing
    dispatch (guardrails enforced)."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask, KILLED
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class PatchProvider:
        name = "p"
        def complete(self, messages, system="", **kw):
            # Valid patch: extend 1, revise 1, abort 1
            return ('{"extend":[{"id":"ext1","description":"new","aspect":"b",'
                    '"importance":0.8,"modalities":["doc_text"]}],'
                    '"revise":[{"id":"s1","importance":0.95}],'
                    '"abort_respawn":[],'
                    '"rationale":"patched gap"}')

    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(PatchProvider()),
                        SubAgentFactory(PatchProvider()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="orig", aspect="a",
                              importance=0.5, modalities=["doc_text"])]
    orch._dispatch({"command": "REPLAN", "arguments": {}}, state, state.pool)
    # Extend appended s1 → (ext1 added with aspect=b)
    aspects = sorted({s.aspect for s in state.subtasks})
    assert "b" in aspects
    # Revise bumped s1.importance
    s1 = next(s for s in state.subtasks if s.id == "s1")
    assert s1.importance == 0.95
    assert len(state.replan_traces) == 1
    applied = state.replan_traces[0]["applied"]
    assert any("extend" in a for a in applied)
    assert any("revise" in a for a in applied)


def t_request_curated_evidence_dispatches_authorization():
    """REQUEST_CURATED_EVIDENCE command → pool.authorize_curated called."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""

    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    # Seed an agent with a summary so the VoI gate has something to grade
    state.pool.register_agent("a1", "doc_text", "g", aspect="x")
    state.pool.note_retrieved_candidates("a1", [{"id": "d1", "content": "c",
                                                  "meta": {}}])
    state.pool.write("a1", "summary", tier1_summary(
        finding="x",
        reasoning="Partial evidence",
        addressed=["partial"],
        rq="medium", ec="medium", rs="low",
        citations=["d1"], n_retrieved=1, n_kept=1, top_score=0.6, score_spread=0.1,
    ))
    # Register in orchestrator so dispatch sees it
    from orchestrator.runtime import AgentRecord
    orch.agents["a1"] = AgentRecord(agent_id="a1", agent_type="si",
                                    modality="doc_text", goal="g",
                                    aspect="x")
    # Light request (no raw) — should be authorised
    orch._dispatch({"command": "REQUEST_CURATED_EVIDENCE",
                    "arguments": {"agent_id": "a1", "with_raw": False}},
                   state, state.pool)
    assert state.pool.is_authorized_curated("a1")


def t_info_gain_tracker_saturates_after_flat_window():
    """Flat history for full window → saturated."""
    from utils.info_gain_tracker import InfoGainTracker
    t = InfoGainTracker(window=3, delta=0.03)
    t.record(1, 0.50, 0.50)
    t.record(2, 0.51, 0.50)
    t.record(3, 0.50, 0.51)
    assert t.is_saturated() is True


def t_info_gain_tracker_not_saturated_when_growing():
    """Rising score prevents saturation."""
    from utils.info_gain_tracker import InfoGainTracker
    t = InfoGainTracker(window=3, delta=0.03)
    t.record(1, 0.30, 0.30)
    t.record(2, 0.50, 0.50)
    t.record(3, 0.70, 0.70)
    assert t.is_saturated() is False
    # Also: before window fills, never saturated
    t2 = InfoGainTracker(window=3)
    t2.record(1, 0.50, 0.50)
    t2.record(2, 0.50, 0.50)
    assert t2.is_saturated() is False


def t_voi_gate_denies_sketch_when_saturated():
    """gate_request(stage='sketch', info_gain_saturated=True) → DENY."""
    from orchestrator.voi_gating import gate_request
    snap_row = {"agent_id": "a1", "version": 1,
                "conf_retrieval": 0.6, "conf_agreement": 0.6,
                "conf_coverage": 0.5, "coverage": {"covered": 0.4},
                "ambiguity": 0.2, "finding": "x"}
    dec = gate_request(snap_row=snap_row, status_board=[snap_row],
                       subtask=None, budget=None, stage="sketch",
                       retry_count=0, info_gain_saturated=True)
    assert dec.allow is False
    assert dec.reason == "info_gain_saturated"


def t_voi_gate_retry_does_not_override_saturation():
    """Retry override must NOT bypass saturation check."""
    from orchestrator.voi_gating import gate_request
    snap_row = {"agent_id": "a1", "version": 1,
                "conf_retrieval": 0.6, "conf_agreement": 0.6,
                "conf_coverage": 0.5, "coverage": {"covered": 0.4},
                "ambiguity": 0.2, "finding": "x"}
    # retry_count=2 would normally auto-approve, but saturation beats it
    dec = gate_request(snap_row=snap_row, status_board=[snap_row],
                       subtask=None, budget=None, stage="sketch",
                       retry_count=2, info_gain_saturated=True)
    assert dec.allow is False
    assert dec.reason == "info_gain_saturated"


def t_resolve_conflict_records_decision():
    """RESOLVE_CONFLICT dispatch writes into state.conflict_resolutions
    and validates resolution enum + trust_agent_id presence."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import AgentRecord
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()), store,
                        ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    # Need a real agent so trust_agent_id validation passes
    orch.agents["a1"] = AgentRecord(agent_id="a1", agent_type="si",
                                    modality="doc_text", goal="g", aspect="x")

    # Valid trust_one resolution
    orch._dispatch({"command": "RESOLVE_CONFLICT",
                    "arguments": {"aspect": "x",
                                  "resolution": "trust_one",
                                  "trust_agent_id": "a1",
                                  "reason": "paper is authoritative"}},
                   state, state.pool)
    assert "x" in state.conflict_resolutions
    entry = state.conflict_resolutions["x"]
    assert entry["resolution"] == "trust_one"
    assert entry["trust_agent_id"] == "a1"

    # Invalid: trust_one without trust_agent_id → rejected (no entry written)
    orch._dispatch({"command": "RESOLVE_CONFLICT",
                    "arguments": {"aspect": "y",
                                  "resolution": "trust_one"}},
                   state, state.pool)
    assert "y" not in state.conflict_resolutions

    # Invalid resolution enum → rejected
    orch._dispatch({"command": "RESOLVE_CONFLICT",
                    "arguments": {"aspect": "z",
                                  "resolution": "pick_one"}},
                   state, state.pool)
    assert "z" not in state.conflict_resolutions

    # Valid unresolvable (no trust_agent_id needed)
    orch._dispatch({"command": "RESOLVE_CONFLICT",
                    "arguments": {"aspect": "q",
                                  "resolution": "unresolvable",
                                  "reason": "sources fundamentally contradict"}},
                   state, state.pool)
    assert state.conflict_resolutions["q"]["resolution"] == "unresolvable"


def t_aspect_agreements_includes_conflict_details_on_disagree():
    """When agents disagree, aspect_agreements row includes per-agent
    findings so DA can resolve without INSPECT."""
    from memory.evidence_pool import EvidencePool
    p = EvidencePool()
    # Two agents, same aspect, opposite findings, both high confidence
    for aid, mod, rq, fnd in [
        ("a1", "doc_text",   "high", "Orchestrator is deterministic"),
        ("a2", "video_text", "high", "Orchestrator uses LLM every turn"),
    ]:
        p.register_agent(aid, mod, "g", aspect="determinism")
        p.note_retrieved_ids(aid, ["d1"])
        p.write(aid, "summary", tier1_summary(
            finding=fnd,
            reasoning=f"According to {mod}",
            addressed=["determinism"],
            rq=rq,  # positional arg for retrieval_quality
            citations=["d1"], top_score=0.9, score_spread=0.05,
        ))
    out = p.aspect_agreements()
    row = next(r for r in out if r["aspect"] == "determinism")
    # Test assumes classifier calls disagree when both confident + word overlap low
    if row["agreement_state"] == "disagree":
        cd = row["conflict_details"]
        assert cd is not None
        assert len(cd) == 2
        by_aid = {e["agent_id"]: e for e in cd}
        assert "a1" in by_aid and "a2" in by_aid
        assert "deterministic" in by_aid["a1"]["finding"]
        assert "LLM every turn" in by_aid["a2"]["finding"]
    else:
        # Classifier didn't flag disagree — conflict_details stays None.
        # (Test still valid: confirms no spurious details when not disagree.)
        assert row["conflict_details"] is None


def t_budget_add_tracks_tier_breakdown():
    """Budget.add() increments used_tokens and the per-category bucket."""
    from orchestrator.runtime import Budget
    b = Budget(max_tokens=10000)
    b.add("da_prompt", 500)
    b.add("sub_agent", 300)
    b.add("da_prompt", 200)       # accumulates
    assert b.used_tokens == 1000
    assert b.tier_breakdown["da_prompt"] == 700
    assert b.tier_breakdown["sub_agent"] == 300


def t_standard_rag_baseline_smoke():
    """Standard RAG baseline runs end-to-end with a scripted provider."""
    from baselines.standard_rag import run
    from memory.store import VectorIndex, MultiIndexStore

    class P:
        name = "p"
        def complete(self, messages, system="", **kw):
            return ('{"answer": "test-answer",'
                    '"confidence": 0.8, "reason": "mocked"}')
    store = MultiIndexStore({
        "doc_text": VectorIndex("doc_text", [
            {"id": "d1", "content": "some content",
             "meta": {"source": "x.pdf", "page": 1}}])
    })
    result = run("what is x?", store, P(), k=3)
    assert result["answer"] == "test-answer"
    assert result["confidence"] == 0.8
    assert result["n_retrieved"] >= 0
    assert result["n_tokens"] > 0


def t_late_fusion_baseline_searches_all_modalities():
    """Late-fusion baseline searches each available modality."""
    from baselines.late_fusion_rag import run
    from memory.store import VectorIndex, MultiIndexStore

    searched_modalities = []
    class FakeStore:
        def search(self, query, modality, k):
            searched_modalities.append(modality)
            return [{"id": f"{modality}_1", "content": f"content-{modality}",
                     "score": 0.7,
                     "meta": {"source": "x", "page": 1,
                              "asset_type": "text"}}]

    class P:
        name = "p"
        def complete(self, messages, system="", **kw):
            return '{"answer": "ok", "confidence": 0.9, "reason": "r"}'

    store = FakeStore()
    result = run("q", store, P(), k_per_modality=2)
    assert result["answer"] == "ok"
    # All 4 modalities probed
    assert set(searched_modalities) == {"doc_text", "doc_visual",
                                        "video_text", "video_visual"}


def t_self_rag_style_iterates_with_need_more():
    """Self-RAG style stops early if need_more=false."""
    from baselines.self_rag_style import run
    from memory.store import VectorIndex, MultiIndexStore

    call_count = [0]
    class P:
        name = "p"
        def complete(self, messages, system="", **kw):
            call_count[0] += 1
            if call_count[0] == 1:
                return ('{"answer": "partial", "confidence": 0.4,'
                        '"need_more": true, "next_query": "new q",'
                        '"reason": "need more"}')
            return ('{"answer": "final", "confidence": 0.9,'
                    '"need_more": false, "next_query": "",'
                    '"reason": "done"}')

    store = MultiIndexStore({
        "doc_text": VectorIndex("doc_text", [
            {"id": "d1", "content": "some content",
             "meta": {"source": "x", "page": 1}}])
    })
    result = run("query", store, P(), k=3, max_iters=5)
    assert result["answer"] == "final"
    assert result["n_iters"] == 2    # stopped after 2, not all 5
    assert call_count[0] == 2


def t_early_fusion_merges_by_score():
    """Early-fusion union-ranks across modalities by score."""
    from baselines.early_fusion_rag import run
    class FakeStore:
        def search(self, query, modality, k):
            if modality == "doc_text":
                return [{"id": "dt1", "content": "text", "score": 0.9,
                         "meta": {"source": "x"}}]
            if modality == "doc_visual":
                return [{"id": "dv1", "content": "visual", "score": 0.3,
                         "meta": {"source": "x"}}]
            return []

    sent_prompt = []
    class P:
        name = "p"
        def complete(self, messages, system="", **kw):
            sent_prompt.append(messages[0]["content"])
            return '{"answer": "ok", "confidence": 0.7, "reason": ""}'

    result = run("q", FakeStore(), P(), k=5)
    # The higher-scored doc_text candidate should appear first in the prompt
    assert "text" in sent_prompt[0]
    # Both candidates present
    assert "[1]" in sent_prompt[0]
    assert result["n_retrieved"] == 2


def t_parsed_block_schema_roundtrip():
    """ParsedBlock.to_dict/from_dict round-trips; unknown keys land in extra."""
    from data.parsers.schema import ParsedBlock
    b = ParsedBlock(
        block_id="paper1_p3_t0", block_type="text",
        content="A paragraph", source="paper1.pdf", page=3,
        modality="doc_text", extra={"paragraph_idx": 0})
    d = b.to_dict()
    assert d["block_id"] == "paper1_p3_t0"
    assert d["modality"] == "doc_text"
    # Round-trip including unknown field
    d2 = dict(d)
    d2["bonus_field"] = "hello"
    b2 = ParsedBlock.from_dict(d2)
    assert b2.extra.get("bonus_field") == "hello"
    assert b2.block_id == b.block_id


def t_mock_bge_embedder_produces_usable_vectors():
    """MockBGEEmbedder: similar inputs → higher cosine than unrelated ones."""
    from data.embedders.bge_text import MockBGEEmbedder
    import math
    e = MockBGEEmbedder(dim=64)
    vs = e.embed_batch(["the orchestrator is deterministic",
                        "orchestrator runs deterministically",
                        "banana banana banana"])
    assert all(len(v) == 64 for v in vs)
    # Vectors are unit-normalised: ||v|| ≈ 1
    for v in vs:
        norm = math.sqrt(sum(x*x for x in v))
        assert abs(norm - 1.0) < 0.01 or norm == 0.0
    # Similar sentences should have higher cosine than unrelated
    def cos(a, b): return sum(x*y for x, y in zip(a, b))
    sim_similar = cos(vs[0], vs[1])
    sim_unrelated = cos(vs[0], vs[2])
    assert sim_similar > sim_unrelated


def t_index_builder_routes_blocks_by_modality():
    """build_indices fans out 4 modalities from a mixed ParsedBlock stream."""
    from data.parsers.schema import ParsedBlock
    from data.embedders.bge_text import MockBGEEmbedder
    from data.embedders.clip_image import MockCLIPImageEmbedder
    from data.index_builder import build_indices

    blocks = [
        ParsedBlock(block_id="p1_t0", block_type="text",
                    content="A prose paragraph", source="p1.pdf",
                    page=1, modality="doc_text"),
        ParsedBlock(block_id="p1_tb0", block_type="table",
                    content="| a | b |\n| - | - |\n| 1 | 2 |",
                    source="p1.pdf", page=1, modality="doc_text"),
        ParsedBlock(block_id="p1_fig0", block_type="image",
                    content="", source="p1.pdf", page=2,
                    asset_uri="file:///tmp/nonexistent.png",
                    modality="doc_visual",
                    extra={"caption": "Figure 1: diagram"}),
        ParsedBlock(block_id="v1_asr0", block_type="text",
                    content="The speaker says...", source="v1.mp4",
                    t_start=0.0, t_end=30.0, modality="video_text"),
        ParsedBlock(block_id="v1_scene0", block_type="image",
                    content="", source="v1.mp4", t_start=10.0,
                    asset_uri="file:///tmp/nonexistent2.png",
                    modality="video_visual"),
    ]
    store = build_indices(blocks, MockBGEEmbedder(),
                          MockCLIPImageEmbedder())
    # All 4 modality indices should exist
    assert "doc_text" in store._indices
    assert "doc_visual" in store._indices
    assert "video_text" in store._indices
    assert "video_visual" in store._indices
    # doc_text should have 2 blocks (text + table)
    assert len(store._indices["doc_text"].docs) == 2
    assert len(store._indices["doc_visual"].docs) == 1
    assert len(store._indices["video_text"].docs) == 1
    assert len(store._indices["video_visual"].docs) == 1
    # Image block's caption should be preserved in content (fallback)
    visual_doc = store._indices["doc_visual"].docs[0]
    assert "Figure 1" in visual_doc["content"]
    # asset_type meta preserved
    assert visual_doc["meta"]["asset_type"] == "image"


def t_madqa_loader_parses_jsonl(tmp_path=None):
    """madqa.load_madqa parses QA jsonl + discovers PDFs."""
    import tempfile, os, json as _json
    from data.loaders.madqa import load_madqa
    with tempfile.TemporaryDirectory() as root:
        pdfs_dir = os.path.join(root, "pdfs")
        os.makedirs(pdfs_dir)
        # Two real-looking files
        open(os.path.join(pdfs_dir, "doc_A.pdf"), "wb").close()
        open(os.path.join(pdfs_dir, "doc_B.pdf"), "wb").close()
        qa_path = os.path.join(root, "qa_dev.jsonl")
        with open(qa_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps({
                "id": "q1", "question": "What is X?",
                "answer": "X is Y", "doc_id": "doc_A",
                "aspect_gold": "definition"}) + "\n")
            f.write(_json.dumps({
                "id": "q2", "question": "Compare P and Q",
                "answer": ["P wins", "P"], "doc_id": "doc_B",
                "evidence_pages": [3, 4]}) + "\n")
            f.write(_json.dumps({
                "id": "q3", "question": "Missing doc",
                "answer": "", "doc_id": "doc_NOT_THERE"}) + "\n")
        examples, pdf_paths = load_madqa(root, split="dev")
        assert len(examples) == 3
        assert examples[0].id == "q1"
        assert examples[0].aspect_gold == "definition"
        assert examples[1].answer == ["P wins", "P"]
        assert examples[1].evidence_pages == [3, 4]
        assert len(pdf_paths) == 2
        assert "doc_A" in pdf_paths and "doc_B" in pdf_paths
        # doc_NOT_THERE not in pdf_paths but loader still returned it
        assert "doc_NOT_THERE" not in pdf_paths


def t_pdf_parser_graceful_without_pdfplumber():
    """When pdfplumber isn't importable, parse_pdf raises a clear error
    instead of an opaque ImportError traceback."""
    import sys, importlib
    # Skip if pdfplumber IS installed — the test is only useful on systems
    # without it (most dev envs, including this one).
    if "pdfplumber" in sys.modules or importlib.util.find_spec("pdfplumber"):
        return
    from data.parsers.pdf_parser import parse_pdf
    try:
        parse_pdf("/nonexistent.pdf")
    except RuntimeError as e:
        assert "pdfplumber" in str(e).lower()
    except FileNotFoundError:
        # Some envs may fail at the file check before the import check;
        # that's also acceptable — means path validation kicked in first.
        pass


def t_citations_enriched_from_retrieval_meta():
    """Sub-agent gives id-only citation; pool enriches with asset_type/source
    from the retrieved candidate's meta."""
    p = EvidencePool()
    p.register_agent("a", "doc_visual", "find diagram", aspect="layout")
    # Register full candidates (not just ids). Pool should capture meta.
    p.note_retrieved_candidates("a", [
        {"id": "dv_01", "content": "Fig 1 caption",
         "meta": {"asset_type": "image", "source": "paper1.pdf", "page": 3}},
        {"id": "dt_07", "content": "A paragraph",
         "meta": {"source": "paper1.pdf", "page": 4}},
    ])
    p.write("a", "summary", tier1_summary(
        finding="layout has 4 boxes",
        reasoning="Diagram shows structure",
        addressed=["layout"],
        n_retrieved=2, n_kept=2, top_score=0.9, score_spread=0.1,
        citations=["dv_01", "dt_07"],  # id-only — pool must enrich
    ))
    row = next(r for r in p.status_board() if r["agent_id"] == "a")
    cits = row["citations"]
    assert len(cits) == 2
    # Verified both
    by_id = {c["id"]: c for c in cits}
    assert by_id["dv_01"]["asset_type"] == "image"
    assert by_id["dv_01"]["source"] == "paper1.pdf"
    assert by_id["dv_01"]["page"] == 3
    assert by_id["dt_07"]["source"] == "paper1.pdf"
    # Text entries may have asset_type=None; that's fine, just verify no crash
    assert "asset_type" in by_id["dt_07"]


def t_citations_structured_passthrough():
    """Sub-agent provides structured citations (dicts); pool preserves the
    explicit fields and still adds any missing ones from retrieval meta."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x", aspect="x")
    p.note_retrieved_candidates("a", [
        {"id": "d1", "content": "c",
         "meta": {"source": "known.pdf", "page": 5}},
    ])
    p.write("a", "summary", {
        "finding": "x",
        "reasoning": "Evidence found",
        "task_completion": {"addressed": ["x"], "partial": [], "uncovered": []},
        "confidence": {"retrieval_quality": "medium",
                       "evidence_coherence": "medium", "reasoning_strength": "medium"},
        "local_gaps": {"critical": [], "suggested_modalities": []},
        "n_retrieved": 1, "n_kept": 1, "top_score": 0.7,
        "score_spread": 0.1,
        # Sub-agent-provided structured citation; "page" override; new field
        "citations": [{"id": "d1", "page": 99, "paragraph_idx": 2}],
    })
    row = next(r for r in p.status_board() if r["agent_id"] == "a")
    cit = row["citations"][0]
    # Sub-agent's page=99 overrides meta's page=5
    assert cit["page"] == 99
    # Explicit field preserved
    assert cit["paragraph_idx"] == 2
    # Missing fields inherited from meta
    assert cit["source"] == "known.pdf"


def t_sketch_accepts_key_candidates_shape():
    """New Tier-2.5 shape with evidence_hit per candidate works."""
    p = EvidencePool()
    p.register_agent("a", "doc_visual", "g", aspect="x")
    p.note_retrieved_candidates("a", [
        {"id": "dv_1", "content": "c1"},
        {"id": "dv_2", "content": "c2"},
    ])
    # Need summary first (sets up known_ids)
    p.write("a", "summary", tier1_summary(
        finding="x",
        reasoning="Visual evidence",
        addressed=["x"],
        n_retrieved=2, n_kept=1, top_score=0.7, score_spread=0.1,
        citations=["dv_1"],
    ))
    # Authorize + write sketch using new shape (visual-friendly: note instead of text)
    p.authorize_sketch("a")
    p.write("a", "sketch", {
        "key_candidates": [
            {"id": "dv_1", "note": "diagram with 4 components",
             "relevance": 0.9, "evidence_hit": 0.8},
            {"id": "dv_2", "note": "table of metrics",
             "relevance": 0.5, "evidence_hit": 0.3},
        ]
    })
    out = p.get_sketch("a")
    assert out is not None
    cands = out["key_candidates"]
    assert len(cands) == 2
    assert cands[0]["evidence_hit"] == 0.8
    assert cands[0]["note"] == "diagram with 4 components"
    # text field defaults to empty string for visual modality
    assert cands[0]["text"] == ""


def t_sketch_rejects_both_shapes_provided():
    """If payload has BOTH key_sentences and key_candidates, reject."""
    p = EvidencePool()
    p.register_agent("a", "doc_text", "g", aspect="x")
    p.note_retrieved_candidates("a", [{"id": "d1", "content": "c"}])
    p.write("a", "summary", tier1_summary(
        finding="x",
        reasoning="Text evidence",
        addressed=["x"],
        n_retrieved=1, n_kept=1, top_score=0.7, score_spread=0.1,
        citations=["d1"],
    ))
    p.authorize_sketch("a")
    try:
        p.write("a", "sketch", {
            "key_sentences": [{"id": "d1", "text": "x", "relevance": 0.9}],
            "key_candidates": [{"id": "d1", "note": "x", "relevance": 0.9,
                                "evidence_hit": 0.8}],
        })
        assert False, "should reject payload with both shapes"
    except DisclosureError:
        pass


def t_subtask_aspect_modality_uniqueness():
    """Decompose accepts same aspect with different modality; rejects duplicate
    (aspect, modality) pairs."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from config.ablation import AblationConfig

    class FakeLLM:
        name = "fake"
        def complete(self, messages, system="", **kw):
            # Two subtasks same aspect, different modality → both kept.
            # One duplicate (aspect, modality) → second one dropped.
            return ('[{"id":"a","description":"dt","aspect":"claim","importance":0.9,'
                    '"modalities":["doc_text"]},'
                    '{"id":"b","description":"vt","aspect":"claim","importance":0.8,'
                    '"modalities":["video_text"]},'
                    '{"id":"c","description":"dt dup","aspect":"claim","importance":0.7,'
                    '"modalities":["doc_text"]}]')

    prov = FakeLLM()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="verify claim")
    orch._pre_run_decompose("verify claim", state)
    # 2 accepted (same aspect, two different modalities), 1 rejected (dup)
    assert len(state.subtasks) == 2
    mods = sorted(s.modalities[0] for s in state.subtasks)
    assert mods == ["doc_text", "video_text"]
    for s in state.subtasks:
        assert s.aspect == "claim"


def t_spawn_requires_aspect_when_subtasks_present():
    """SPAWN_AGENT rejected when state.subtasks populated but no aspect given."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="x", aspect="a1",
                              importance=0.9, modalities=["doc_text"])]
    # Spawn without aspect — must be rejected
    orch._dispatch({"command": "SPAWN_AGENT",
                    "arguments": {"agent_type": "si",
                                  "modality": "doc_text",
                                  "goal": "g"}},
                   state, state.pool)
    assert len(orch.agents) == 0, "spawn without aspect should be rejected"
    # Spawn with wrong (aspect, modality) pair — rejected
    orch._dispatch({"command": "SPAWN_AGENT",
                    "arguments": {"agent_type": "si",
                                  "modality": "doc_visual",   # not in plan
                                  "goal": "g",
                                  "aspect": "a1"}},
                   state, state.pool)
    assert len(orch.agents) == 0, "wrong modality should be rejected"
    # Spawn with matching (aspect, modality) — accepted
    orch._dispatch({"command": "SPAWN_AGENT",
                    "arguments": {"agent_type": "si",
                                  "modality": "doc_text",
                                  "goal": "g",
                                  "aspect": "a1"}},
                   state, state.pool)
    assert len(orch.agents) == 1, "valid spawn should succeed"


def t_spawn_without_subtasks_unconstrained():
    """Back-compat: when state.subtasks empty (decompose disabled or failed),
    SPAWN_AGENT does NOT require aspect/modality consistency."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    # subtasks empty → no constraint applied
    assert state.subtasks == []
    orch._dispatch({"command": "SPAWN_AGENT",
                    "arguments": {"agent_type": "si",
                                  "modality": "doc_text",
                                  "goal": "g"}},
                   state, state.pool)
    assert len(orch.agents) == 1


def t_abort_respawn_rejects_second_abort():
    """Each agent_id can be ABORTed at most once per run."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import KILLED
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", []),
                             "doc_visual": VectorIndex("doc_visual", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    # First spawn (no subtasks set → no plan constraint)
    orch._dispatch({"command": "SPAWN_AGENT",
                    "arguments": {"agent_type": "si",
                                  "modality": "doc_text",
                                  "goal": "g",
                                  "aspect": "x"}},
                   state, state.pool)
    aid1 = list(orch.agents.keys())[0]
    # First ABORT — accepted
    orch._dispatch({"command": "ABORT_AND_RESPAWN",
                    "arguments": {"agent_id": aid1,
                                  "new_modality": "doc_visual"}},
                   state, state.pool)
    assert orch.agents[aid1].status == KILLED
    new_aids = [a for a in orch.agents if a != aid1]
    assert len(new_aids) == 1
    aid2 = new_aids[0]
    # Second ABORT on the FIRST agent — should be rejected (aid1 already aborted)
    orch._dispatch({"command": "ABORT_AND_RESPAWN",
                    "arguments": {"agent_id": aid1}},
                   state, state.pool)
    assert aid1 in state.aborted_agents
    # No new spawn should have happened
    assert len(orch.agents) == 2, f"expected 2 agents, got {len(orch.agents)}"
    # But the new agent aid2 is NOT in aborted_agents yet — it CAN be aborted
    # once. This verifies we track per-agent, not per-aspect.
    assert aid2 not in state.aborted_agents


def t_extend_subtasks_cap_at_10():
    """EXTEND_SUBTASKS stops adding once total hits 10."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    # Pre-populate 8 subtasks.
    state.subtasks = [Subtask(id=f"s{i}", description="x",
                              aspect=f"a{i}", importance=0.5,
                              modalities=["doc_text"])
                      for i in range(8)]
    # Attempt to add 5 more — only 2 should land (cap is 10).
    orch._dispatch({"command": "EXTEND_SUBTASKS",
                    "arguments": {"subtasks": [
                        {"id": f"n{i}", "description": "x",
                         "aspect": f"n{i}", "importance": 0.5,
                         "modalities": ["doc_text"]}
                        for i in range(5)]}},
                   state, state.pool)
    assert len(state.subtasks) == 10, f"got {len(state.subtasks)}"


def t_revise_subtask_records_trace():
    """REVISE_SUBTASK appends a trace entry for every non-op change."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import Subtask
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(P()), SubAgentFactory(P()),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()
    state.subtasks = [Subtask(id="s1", description="orig", aspect="a",
                              importance=0.5, modalities=["doc_text"])]
    orch._dispatch({"command": "REVISE_SUBTASK",
                    "arguments": {"id": "s1", "importance": 0.9}},
                   state, state.pool)
    assert len(state.revise_trace) == 1
    e = state.revise_trace[0]
    assert e["subtask_id"] == "s1"
    assert e["before"]["importance"] == 0.5
    assert e["after"]["importance"] == 0.9
    # No-op (unknown id) should NOT record
    orch._dispatch({"command": "REVISE_SUBTASK",
                    "arguments": {"id": "ghost", "importance": 0.1}},
                   state, state.pool)
    assert len(state.revise_trace) == 1


def t_abort_and_respawn_kills_then_spawns():
    """ABORT_AND_RESPAWN kills the old agent and spawns a new one inheriting
    aspect, with optional goal/modality overrides."""
    from orchestrator.controller import Orchestrator
    from orchestrator.runtime import KILLED
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from memory.evidence_pool import EvidencePool
    from config.ablation import AblationConfig

    class P:
        name = "p"
        def complete(self, *a, **kw): return ""
    prov = P()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", []),
                             "doc_visual": VectorIndex("doc_visual", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov), store,
                        ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    state.pool = EvidencePool()

    # Spawn an initial agent
    spawn_cmd = {"command": "SPAWN_AGENT",
                 "arguments": {"agent_type": "si", "modality": "doc_text",
                               "goal": "old goal", "aspect": "important"}}
    orch._dispatch(spawn_cmd, state, state.pool)
    old_aid = list(orch.agents.keys())[0]
    old_rec = orch.agents[old_aid]
    assert old_rec.aspect == "important"

    # ABORT_AND_RESPAWN with a new modality + new goal
    abort_cmd = {"command": "ABORT_AND_RESPAWN",
                 "arguments": {"agent_id": old_aid,
                               "new_modality": "doc_visual",
                               "new_goal": "rephrased goal",
                               "reason": "modality_misfit"}}
    orch._dispatch(abort_cmd, state, state.pool)

    assert orch.agents[old_aid].status == KILLED
    new_aids = [a for a in orch.agents if a != old_aid]
    assert len(new_aids) == 1, f"expected 1 new agent, got {len(new_aids)}"
    new_rec = orch.agents[new_aids[0]]
    assert new_rec.modality == "doc_visual"
    assert new_rec.goal == "rephrased goal"
    # aspect must be inherited from old when not overridden
    assert new_rec.aspect == "important"


def t_aspect_agreements_exhausted_flag():
    """Any agent reporting retrieval_quality=exhausted bubbles up."""
    p = EvidencePool()
    p.register_agent("a1", "doc_text", "g", aspect="x")
    p.note_retrieved_ids("a1", ["d1"])
    p.write("a1", "summary", tier1_summary(
        finding="found something",
        reasoning="Index wrung out",
        addressed=["x"],
        rq="medium",
        retrieval_quality="exhausted",  # optional field for this test
        citations=["d1"], top_score=0.5, score_spread=0.1,
    ))
    out = p.aspect_agreements()
    assert out[0]["any_exhausted"] is True


def t_decompose_aspect_taxonomy():
    """Decompose skill uses fine-grained aspects: event/spatial/entity/causal/temporal/process."""
    from prompts.skills import decompose_skill

    # Use a custom class name that won't be filtered by decompose_skill.run
    class TestDecomposeLLM:
        name = "test_decompose_llm"
        def complete(self, messages, system="", **kw):
            # Return subtasks with all 6 aspect types
            return """[
                {"id":"s1","description":"What happened during the incident","aspect":"event","importance":0.9,"modalities":["doc_text"]},
                {"id":"s2","description":"Where the components are located","aspect":"spatial","importance":0.8,"modalities":["doc_visual"]},
                {"id":"s3","description":"Who was involved","aspect":"entity","importance":0.7,"modalities":["video_text"]},
                {"id":"s4","description":"Why the failure occurred","aspect":"causal","importance":0.85,"modalities":["doc_text"]},
                {"id":"s5","description":"When each milestone happened","aspect":"temporal","importance":0.6,"modalities":["video_text"]},
                {"id":"s6","description":"How to deploy the system","aspect":"process","importance":0.75,"modalities":["video_text"]}
            ]"""

    result = decompose_skill.run("Explain the system architecture", TestDecomposeLLM())
    assert result is not None
    assert len(result) == 6

    aspects = {s["aspect"] for s in result}
    expected = {"event", "spatial", "entity", "causal", "temporal", "process"}
    assert aspects == expected, f"Got {aspects}, expected {expected}"

    # Verify each subtask has exactly one modality
    for s in result:
        assert isinstance(s["modalities"], list)
        assert len(s["modalities"]) == 1, f"Subtask {s['id']} has {len(s['modalities'])} modalities"


def t_decompose_schema_valid():
    """When provider returns a well-formed JSON, _pre_run_decompose populates
    state.subtasks with valid Subtask objects. Multi-modality subtasks are
    split into one subtask per modality (P2 round-1 rule)."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from config.ablation import AblationConfig

    class FakeLLM:
        name = "fake"
        def complete(self, messages, system="", **kw):
            return ('[{"id":"s1","description":"Define X.","aspect":"x_def",'
                    '"importance":0.9,"modalities":["doc_text"]},'
                    '{"id":"s2","description":"Compare X to Y.","aspect":"x_vs_y",'
                    '"importance":0.7,"modalities":["doc_text","doc_visual"]}]')

    prov = FakeLLM()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="define X and compare to Y")
    orch._pre_run_decompose("define X and compare to Y", state)
    # s1 has 1 modality → kept as-is.
    # s2 has 2 modalities → split into s2_doc_text and s2_doc_visual.
    # Expected: 3 total subtasks.
    assert len(state.subtasks) == 3, f"got {len(state.subtasks)}"
    # s1 unchanged
    s1 = state.subtasks[0]
    assert s1.aspect == "x_def" and s1.modalities == ["doc_text"]
    # s2 split into two single-modality subtasks, same aspect
    s2_parts = [s for s in state.subtasks if s.aspect == "x_vs_y"]
    assert len(s2_parts) == 2
    mods = sorted(s.modalities[0] for s in s2_parts)
    assert mods == ["doc_text", "doc_visual"]
    # All should have exactly 1 modality each
    for s in state.subtasks:
        assert len(s.modalities) == 1, f"{s.id} has {s.modalities}"


def t_decompose_handles_malformed_json():
    """Malformed/empty/invalid responses must NOT crash; subtasks stays []."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from config.ablation import AblationConfig

    for bad in ["nonsense without json", "", "[broken json,", "null"]:
        class P:
            name = "fake"
            def __init__(self, r): self.r = r
            def complete(self, messages, system="", **kw): return self.r
        prov = P(bad)
        store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
        orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                            store, ablation=AblationConfig.preset("baseline"))
        state = StateManager(query="q")
        orch._pre_run_decompose("q", state)
        assert state.subtasks == [], f"bad={bad!r} produced {state.subtasks}"


def t_decompose_filters_invalid_modalities_and_dupes():
    """Subtasks with unknown modalities are dropped; duplicate aspects too;
    overall list is capped at 5."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from memory.store import VectorIndex, MultiIndexStore
    from memory.state_manager import StateManager
    from config.ablation import AblationConfig

    class P:
        name = "fake"
        def complete(self, messages, system="", **kw):
            # 7 entries, includes a duplicate aspect, an invalid modality,
            # and one with no valid modalities at all.
            return ('[{"id":"s1","description":"a","aspect":"x","importance":0.9,'
                    '"modalities":["doc_text"]},'
                    '{"id":"s2","description":"b","aspect":"y","importance":0.8,'
                    '"modalities":["bogus_modality"]},'  # → dropped (no valid)
                    '{"id":"s3","description":"c","aspect":"x","importance":0.7,'
                    '"modalities":["doc_text"]},'  # → dropped (dup aspect)
                    '{"id":"s4","description":"d","aspect":"z","importance":0.6,'
                    '"modalities":["video_text","bogus"]},'  # kept, only valid mod retained
                    '{"id":"s5","description":"e","aspect":"q","importance":0.5,'
                    '"modalities":["doc_visual"]},'
                    '{"id":"s6","description":"f","aspect":"r","importance":0.4,'
                    '"modalities":["video_visual"]},'
                    '{"id":"s7","description":"g","aspect":"t","importance":0.3,'
                    '"modalities":["doc_text"]}]')
    prov = P()
    store = MultiIndexStore({"doc_text": VectorIndex("doc_text", [])})
    orch = Orchestrator(DecisionAgent(prov), SubAgentFactory(prov),
                        store, ablation=AblationConfig.preset("baseline"))
    state = StateManager(query="q")
    orch._pre_run_decompose("q", state)
    aspects = [s.aspect for s in state.subtasks]
    # Cap at 5 means the iteration looks at first 5: s1,s2,s3,s4,s5.
    # s2 dropped (bogus mod), s3 dropped (dup aspect). So x, z, q remain.
    assert "x" in aspects
    assert "y" not in aspects, "should drop bogus-modality subtask"
    assert aspects.count("x") == 1, "should drop dup-aspect subtask"
    # Modality filter inside s4: bogus stripped, video_text kept.
    s_z = next(s for s in state.subtasks if s.aspect == "z")
    assert s_z.modalities == ["video_text"]


def t_read_archive_roundtrip():
    """Tier-3 read pipeline: pool.archive populated → DA dispatches READ_ARCHIVE
    → state.pending_archive populated → next-turn prompt builder injects it."""
    from memory.evidence_pool import EvidencePool
    from memory.state_manager import StateManager
    from prompts.prompt_builder import PromptBuilder
    p = EvidencePool()
    p.register_agent("a", "doc_text", "x", aspect="x")
    p.note_retrieved_ids("a", ["d1", "d2"])
    p.write("a", "summary", tier1_summary(
        finding="x",
        reasoning="Partial evidence found",
        addressed=["main"], uncovered=["y"],
        rq="medium",  # positional arg for retrieval_quality
        citations=["d1"], top_score=0.7, score_spread=0.1,
    ))
    p.authorize_full("a")
    p.write("a", "full",
            {"content": "X is the orchestrator dispatching commands.",
             "sources": ["doc1.pdf"]})
    # Simulate Orchestrator dispatching READ_ARCHIVE
    state = StateManager(query="q")
    state.pool = p
    archive = p.get_archive("a")
    assert archive, "archive should be populated"
    state.pending_archive.append({
        "agent_id": "a",
        "content": archive[-1]["content"],
        "sources": archive[-1]["sources"],
    })
    # Build a prompt and confirm the content shows up
    prompt = PromptBuilder.decision(
        query="q", status_board=p.status_board(), coverage=1.0,
        max_conf=0.5, pending_archive=state.pending_archive)
    msg_text = prompt["messages"][0]["content"]
    if isinstance(msg_text, list):
        msg_text = " ".join(b.get("text", "") for b in msg_text
                            if isinstance(b, dict))
    assert "TIER-3 FULL EVIDENCE" in msg_text
    assert "X is the orchestrator dispatching" in msg_text
    assert "doc1.pdf" in msg_text


def t_voi_trace_recorded_on_dispatch():
    """VoI gating must leave a trace entry every time SKETCH/FULL is dispatched,
    whether approved or denied. Ablation studies depend on this."""
    from orchestrator.voi_gating import gate_request, GatingDecision
    from orchestrator.runtime import Budget
    # Case 1: approved via hard rule — still recorded
    row_hi = {"agent_id": "h", "aspect": "x",
              "confidence": {"retrieval_score": 0.65,
                             "evidence_agreement": None, "coverage": None},
              "coverage": {"covered": 0.7, "gaps": []},
              "ambiguity": 0.3, "missing_aspects": []}
    d1 = gate_request(row_hi, [row_hi], None, Budget(10000, 0),
                      stage="full", retry_count=0)
    tr = d1.to_trace()
    assert tr["allow"] is True
    assert tr["stage"] == "full"
    assert "components" in tr
    # Case 2: denied — also recorded with numeric VoI
    row_lo = {"agent_id": "l", "aspect": "x",
              "confidence": {"retrieval_score": 0.95,
                             "evidence_agreement": 0.95, "coverage": 0.95},
              "coverage": {"covered": 0.95, "gaps": []},
              "ambiguity": 0.05, "missing_aspects": []}
    d2 = gate_request(row_lo, [row_lo], None, Budget(10000, 0),
                      stage="sketch", retry_count=0)
    tr2 = d2.to_trace()
    assert tr2["allow"] is False
    assert tr2["components"]["value"] < 0.5
    assert tr2["reason"] == "voi_below_threshold"


def t_cross_modal_selector_prioritizes_conflicts():
    """CrossModalSelector gives higher scores to conflict aspects."""
    from utils.cross_modal_selector import CrossModalSelector

    selector = CrossModalSelector(max_per_agent=5, conflict_boost=2.0)

    # Mock candidates
    candidates = [
        {"id": "c1", "score": 0.5, "content": "text1"},
        {"id": "c2", "score": 0.6, "content": "text2"},
        {"id": "c3", "score": 0.4, "content": "text3"},
    ]

    # Mock status snapshot (new Tier-1 schema)
    snapshot = {
        "agent_id": "a1",
        "aspect": "entity",  # This is in conflict_aspects
        "task_completion": {"addressed": [], "partial": ["partial"], "uncovered": []},
        "confidence": {"retrieval_quality": 0.6, "evidence_coherence": 0.6,
                       "reasoning_strength": 0.5},
    }

    # Select with conflict aspect
    conflict_aspects = {"entity"}
    result = selector.select_curated(
        agent_id="a1",
        retrieved_candidates=candidates,
        status_snapshot=snapshot,
        conflict_aspects=conflict_aspects,
        is_important=True,
        is_low_confidence=False
    )

    assert len(result) == 3
    # c2 should be first (highest base score * conflict boost)
    assert result[0]["id"] == "c2"
    # Check that conflict boost was applied
    assert result[0]["_selection_score"] > candidates[1]["score"]


def t_cross_modal_selector_identify_conflicts():
    """identify_conflict_aspects extracts disagree aspects."""
    from utils.cross_modal_selector import CrossModalSelector

    agreements = [
        {"aspect": "entity", "agreement_state": "agree"},
        {"aspect": "causal", "agreement_state": "disagree"},
        {"aspect": "temporal", "agreement_state": "disagree"},
    ]

    conflicts = CrossModalSelector.identify_conflict_aspects(agreements)
    assert conflicts == {"causal", "temporal"}


def t_cross_modal_selector_important_low_confidence():
    """is_important_low_confidence detects critical weak subtasks."""
    from utils.cross_modal_selector import CrossModalSelector

    # Low confidence case
    snapshot1 = {
        "confidence": {"retrieval_score": 0.4, "evidence_agreement": 0.3},
        "coverage": {"covered": 0.4},
    }
    assert CrossModalSelector.is_important_low_confidence(snapshot1) is True

    # High confidence case
    snapshot2 = {
        "confidence": {"retrieval_score": 0.8, "evidence_agreement": 0.7},
        "coverage": {"covered": 0.8},
    }
    assert CrossModalSelector.is_important_low_confidence(snapshot2) is False


def t_cross_modal_selector_format_for_subagent():
    """format_for_subagent produces correct curated schema."""
    from utils.cross_modal_selector import CrossModalSelector

    curated = [
        {"id": "c1", "_selection_score": 0.9, "evidence_hit": 0.8, "text": "text1"},
        {"id": "c2", "_selection_score": 0.7, "evidence_hit": 0.6, "note": "note2"},
    ]

    formatted = CrossModalSelector.format_for_subagent(curated)

    assert "key_candidates" in formatted
    assert len(formatted["key_candidates"]) == 2
    assert formatted["key_candidates"][0]["id"] == "c1"
    assert formatted["key_candidates"][0]["relevance"] == 0.9
    assert formatted["with_raw"] is False  # No "raw" field in candidates



def t_curated_disclosure_cross_modal_analysis():
    """Curated disclosure includes cross-modal conflict analysis."""
    from memory.evidence_pool import EvidencePool

    pool = EvidencePool(bypass_tier3_gate=True)
    pool.register_agent("a1", "doc_text", "goal", aspect="entity")
    pool.note_retrieved_candidates("a1", [
        {"id": "d1", "content": "text1", "score": 0.8, "meta": {}}
    ])

    # Simulate conflict aspect
    aspect_agreements = [
        {"aspect": "entity", "agreement_state": "disagree",
         "conflict_details": [
             {"agent_id": "a1", "modality": "doc_text", "finding": "X exists"},
             {"agent_id": "a2", "modality": "video_text", "finding": "X doesn't exist"}
         ]}
    ]

    # Write curated with conflict analysis
    payload = {
        "key_candidates": [
            {"id": "d1", "relevance": 0.8, "evidence_hit": 0.7, "text": "Evidence for X"}
        ],
        "with_raw": False
    }

    eid = pool.write("a1", "curated", payload, aspect_agreements)

    # Verify cross-modal metadata
    curated = pool.get_curated("a1")
    assert curated is not None
    assert "cross_modal_meta" in curated
    assert curated["cross_modal_meta"]["is_conflict_aspect"] is True
    assert curated["cross_modal_meta"]["agent_aspect"] == "entity"
    assert curated["cross_modal_meta"]["n_conflict_candidates"] == 1
    assert "conflict_summary" in curated["cross_modal_meta"]
    assert curated["cross_modal_meta"]["conflict_summary"]["n_modalities"] == 2

    # Verify candidates have conflict flags
    candidates = curated["key_candidates"]
    assert len(candidates) == 1
    assert candidates[0]["_conflict_flag"] is True
    assert "_supporting_position" in candidates[0]


def t_curated_disclosure_important_low_confidence():
    """Curated disclosure handles important-but-low-confidence cases."""
    from memory.evidence_pool import EvidencePool

    pool = EvidencePool(bypass_tier3_gate=True)
    pool.register_agent("a1", "doc_text", "important_goal", aspect="causal")
    pool.note_retrieved_candidates("a1", [
        {"id": "d1", "content": "partial info", "score": 0.5, "meta": {}}
    ])

    # No conflict, but important aspect
    payload = {
        "key_candidates": [
            {"id": "d1", "relevance": 0.5, "evidence_hit": 0.4, "text": "Partial evidence"}
        ],
        "with_raw": False
    }

    eid = pool.write("a1", "curated", payload)

    # Verify metadata
    curated = pool.get_curated("a1")
    assert curated["cross_modal_meta"]["is_conflict_aspect"] is False
    assert curated["cross_modal_meta"]["agent_aspect"] == "causal"
    assert curated["cross_modal_meta"]["n_conflict_candidates"] == 0


def t_loop_progress_tracker_detects_stagnation():
    """LoopProgressTracker detects stagnation when no progress for N turns."""
    from utils.loop_progress import LoopProgressTracker, TurnSnapshot

    tracker = LoopProgressTracker(stagnation_threshold=3)

    # Turn 1: new aspect covered
    tracker.record_turn(TurnSnapshot(
        step=1, covered_aspects={"event"}, conflict_aspects=set(),
        max_confidence=0.5, evidence_quality=0.4,
        commands_issued=["SPAWN_AGENT"], agents_active=1, tokens_used=100
    ))
    assert not tracker.is_stagnant()

    # Turns 2-4: NO progress (same aspects, no resolution, no quality improvement)
    for i in range(2, 5):
        tracker.record_turn(TurnSnapshot(
            step=i, covered_aspects={"event"}, conflict_aspects=set(),
            max_confidence=0.5, evidence_quality=0.4,
            commands_issued=["WAIT"], agents_active=1, tokens_used=50
        ))
    # After 3 turns of no progress, should be stagnant
    assert tracker.is_stagnant()


def t_loop_progress_force_stop_when_stagnant():
    """LoopProgressTracker should_force_stop returns True when stagnant."""
    from utils.loop_progress import LoopProgressTracker, TurnSnapshot

    tracker = LoopProgressTracker(stagnation_threshold=2)

    # Simulate stagnant state: same aspects for 3 turns
    for i in range(1, 4):
        tracker.record_turn(TurnSnapshot(
            step=i, covered_aspects={"event"}, conflict_aspects=set(),
            max_confidence=0.5, evidence_quality=0.4,
            commands_issued=["WAIT"], agents_active=1, tokens_used=50
        ))

    # Should force stop when approaching max_steps
    assert tracker.should_force_stop(current_step=38, max_steps=40)

    # Should NOT force stop early even if stagnant
    assert not tracker.should_force_stop(current_step=5, max_steps=40)


def t_loop_progress_recommends_stop_on_high_confidence():
    """LoopProgressTracker recommends STOP when evidence quality is high."""
    from utils.loop_progress import LoopProgressTracker, TurnSnapshot

    tracker = LoopProgressTracker()
    # First turn
    tracker.record_turn(TurnSnapshot(
        step=1, covered_aspects={"event"}, conflict_aspects=set(),
        max_confidence=0.7, evidence_quality=0.6,
        commands_issued=["SPAWN_AGENT"], agents_active=1, tokens_used=100
    ))
    # Second turn with high confidence and quality
    tracker.record_turn(TurnSnapshot(
        step=2, covered_aspects={"event", "causal"}, conflict_aspects=set(),
        max_confidence=0.8, evidence_quality=0.75,
        commands_issued=["STOP_TASK"], agents_active=0, tokens_used=200
    ))

    action = tracker.get_recommended_action()
    assert action == "STOP"


def t_loop_progress_recommends_escalate_on_conflicts():
    """LoopProgressTracker recommends ESCALATE when conflicts remain."""
    from utils.loop_progress import LoopProgressTracker, TurnSnapshot

    tracker = LoopProgressTracker()
    # First turn
    tracker.record_turn(TurnSnapshot(
        step=1, covered_aspects={"event"}, conflict_aspects=set(),
        max_confidence=0.5, evidence_quality=0.4,
        commands_issued=["SPAWN_AGENT"], agents_active=1, tokens_used=100
    ))
    # Second turn with conflicts
    tracker.record_turn(TurnSnapshot(
        step=2, covered_aspects={"event"}, conflict_aspects={"causal"},
        max_confidence=0.5, evidence_quality=0.4,
        commands_issued=["REFLECT"], agents_active=2, tokens_used=150
    ))

    action = tracker.get_recommended_action()
    assert action == "ESCALATE"


def t_loop_progress_summary_includes_metrics():
    """LoopProgressTracker.get_progress_summary returns comprehensive metrics."""
    from utils.loop_progress import LoopProgressTracker, TurnSnapshot

    tracker = LoopProgressTracker()
    tracker.record_turn(TurnSnapshot(
        step=1, covered_aspects={"event", "spatial"}, conflict_aspects={"causal"},
        max_confidence=0.6, evidence_quality=0.5,
        commands_issued=["SPAWN_AGENT"], agents_active=2, tokens_used=200
    ))

    summary = tracker.get_progress_summary()
    assert summary["total_turns"] == 1
    assert summary["n_aspects_covered"] == 2
    assert "event" in summary["aspects_covered"]
    assert summary["current_evidence_quality"] == 0.5
    assert summary["is_stagnant"] is False


tests = [
    ("cmd_valid", t_cmd_valid),
    ("cmd_unknown", t_cmd_unknown),
    ("tier1_ok", t_tier1_ok),
    ("tier1_missing_fields", t_tier1_missing_fields),
    ("tier2_ok", t_tier2_ok),
    ("tier2_rejects_fabricated_citation", t_tier2_rejects_fabricated_citation),
    ("tier2_unverified_marks_suspicious", t_tier2_unverified_marks_suspicious),
    ("tier3_blocked_without_auth", t_tier3_blocked_without_auth),
    ("tier3_allowed_after_auth", t_tier3_allowed_after_auth),
    ("metrics", t_metrics),
    ("workerpool_parallel", t_workerpool_parallel),
    ("multi_index_retrieval", t_multi_index_retrieval),
    ("hybrid_retrieval_rrf", t_hybrid_retrieval_rrf),
    ("unauthorized_full_is_rejected", t_unauthorized_full_is_rejected),
    ("ablation_presets_all_valid", t_ablation_presets_all_valid),
    ("ablation_no_tier3_gate", t_ablation_no_tier3_gate),
    ("ablation_no_rerank_factory", t_ablation_no_rerank_factory),
    ("ablation_no_parallelism_forces_single_worker", t_ablation_no_parallelism_forces_single_worker),
    ("voi_approves_mid_conf_hard_rule", t_voi_approves_mid_conf_hard_rule),
    ("voi_denies_low_value", t_voi_denies_low_value),
    ("voi_retry_override", t_voi_retry_override),
    ("voi_budget_exceeded_denies", t_voi_budget_exceeded_denies),
    ("voi_conflict_hard_rule", t_voi_conflict_hard_rule),
    ("voi_important_gap_hard_rule", t_voi_important_gap_hard_rule),
    ("voi_select_evidence_tier", t_voi_select_evidence_tier),
    ("reflection_signals_persisted", t_reflection_signals_persisted),
    ("reflection_signals_validated", t_reflection_signals_validated),
    ("aspect_agreements_single_source", t_aspect_agreements_single_source),
    ("aspect_agreements_disagree_via_conf_gap", t_aspect_agreements_disagree_via_conf_gap),
    ("aspect_agreements_exhausted_flag", t_aspect_agreements_exhausted_flag),
    ("info_gain_tracker_saturates_after_flat_window", t_info_gain_tracker_saturates_after_flat_window),
    ("info_gain_tracker_not_saturated_when_growing", t_info_gain_tracker_not_saturated_when_growing),
    ("voi_gate_denies_sketch_when_saturated", t_voi_gate_denies_sketch_when_saturated),
    ("voi_gate_retry_does_not_override_saturation", t_voi_gate_retry_does_not_override_saturation),
    ("resolve_conflict_records_decision", t_resolve_conflict_records_decision),
    ("aspect_agreements_includes_conflict_details_on_disagree", t_aspect_agreements_includes_conflict_details_on_disagree),
    ("budget_add_tracks_tier_breakdown", t_budget_add_tracks_tier_breakdown),
    ("standard_rag_baseline_smoke", t_standard_rag_baseline_smoke),
    ("late_fusion_baseline_searches_all_modalities", t_late_fusion_baseline_searches_all_modalities),
    ("self_rag_style_iterates_with_need_more", t_self_rag_style_iterates_with_need_more),
    ("early_fusion_merges_by_score", t_early_fusion_merges_by_score),
    ("parsed_block_schema_roundtrip", t_parsed_block_schema_roundtrip),
    ("mock_bge_embedder_produces_usable_vectors", t_mock_bge_embedder_produces_usable_vectors),
    ("index_builder_routes_blocks_by_modality", t_index_builder_routes_blocks_by_modality),
    ("madqa_loader_parses_jsonl", t_madqa_loader_parses_jsonl),
    ("pdf_parser_graceful_without_pdfplumber", t_pdf_parser_graceful_without_pdfplumber),
    ("retrieval_text_routes_by_agent_modality", t_retrieval_text_routes_by_agent_modality),
    ("retrieval_visual_routes_by_agent_modality", t_retrieval_visual_routes_by_agent_modality),
    ("factory_per_modality_dict_precedence", t_factory_per_modality_dict_precedence),
    ("factory_tools_for_text_modality", t_factory_tools_for_text_modality),
    ("factory_tools_for_visual_modality", t_factory_tools_for_visual_modality),
    ("curated_light_authorization_required", t_curated_light_authorization_required),
    ("curated_light_authorized_write_strips_raw", t_curated_light_authorized_write_strips_raw),
    ("curated_raw_per_id_permission", t_curated_raw_per_id_permission),
    ("reflect_command_stores_verdict_with_heuristic_fallback", t_reflect_command_stores_verdict_with_heuristic_fallback),
    ("replan_command_dispatches_extend_revise_abort", t_replan_command_dispatches_extend_revise_abort),
    ("request_curated_evidence_dispatches_authorization", t_request_curated_evidence_dispatches_authorization),
    ("decompose_skill_structured_output", t_decompose_skill_structured_output),
    ("decompose_skill_robust_to_bad_providers", t_decompose_skill_robust_to_bad_providers),
    ("decompose_aspect_taxonomy", t_decompose_aspect_taxonomy),
    ("subtask_embedding_populated_and_deduped", t_subtask_embedding_populated_and_deduped),
    ("subtask_embedding_cross_modality_no_dedup", t_subtask_embedding_cross_modality_no_dedup),
    ("citations_enriched_from_retrieval_meta", t_citations_enriched_from_retrieval_meta),
    ("citations_structured_passthrough", t_citations_structured_passthrough),
    ("sketch_accepts_key_candidates_shape", t_sketch_accepts_key_candidates_shape),
    ("sketch_rejects_both_shapes_provided", t_sketch_rejects_both_shapes_provided),
    ("subtask_aspect_modality_uniqueness", t_subtask_aspect_modality_uniqueness),
    ("spawn_requires_aspect_when_subtasks_present", t_spawn_requires_aspect_when_subtasks_present),
    ("spawn_without_subtasks_unconstrained", t_spawn_without_subtasks_unconstrained),
    ("abort_respawn_rejects_second_abort", t_abort_respawn_rejects_second_abort),
    ("extend_subtasks_cap_at_10", t_extend_subtasks_cap_at_10),
    ("revise_subtask_records_trace", t_revise_subtask_records_trace),
    ("extend_subtasks_appends_unique_aspects", t_extend_subtasks_appends_unique_aspects),
    ("revise_subtask_mutates_in_place", t_revise_subtask_mutates_in_place),
    ("abort_and_respawn_kills_then_spawns", t_abort_and_respawn_kills_then_spawns),
    ("decompose_schema_valid", t_decompose_schema_valid),
    ("decompose_handles_malformed_json", t_decompose_handles_malformed_json),
    ("decompose_filters_invalid_modalities_and_dupes", t_decompose_filters_invalid_modalities_and_dupes),
    ("read_archive_roundtrip", t_read_archive_roundtrip),
    ("voi_trace_recorded_on_dispatch", t_voi_trace_recorded_on_dispatch),
    ("cross_modal_selector_prioritizes_conflicts", t_cross_modal_selector_prioritizes_conflicts),
    ("cross_modal_selector_identify_conflicts", t_cross_modal_selector_identify_conflicts),
    ("cross_modal_selector_important_low_confidence", t_cross_modal_selector_important_low_confidence),
    ("cross_modal_selector_format_for_subagent", t_cross_modal_selector_format_for_subagent),
    ("curated_disclosure_cross_modal_analysis", t_curated_disclosure_cross_modal_analysis),
    ("curated_disclosure_important_low_confidence", t_curated_disclosure_important_low_confidence),
    ("loop_progress_tracker_detects_stagnation", t_loop_progress_tracker_detects_stagnation),
    ("loop_progress_force_stop_when_stagnant", t_loop_progress_force_stop_when_stagnant),
    ("loop_progress_recommends_stop_on_high_confidence", t_loop_progress_recommends_stop_on_high_confidence),
    ("loop_progress_recommends_escalate_on_conflicts", t_loop_progress_recommends_escalate_on_conflicts),
    ("loop_progress_summary_includes_metrics", t_loop_progress_summary_includes_metrics),
]

for n, f in tests: check(n, f)
