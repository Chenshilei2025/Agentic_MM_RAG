"""REPLAN skill — dynamic subtask reallocation based on gaps.

Core Design for EMNLP paper:
  - Executes AFTER Tier-2 disclosure when REFLECT recommends REPLAN
  - Identifies missing modalities for critical aspects
  - Patches gaps while preserving successful subtasks
  - Ensures each iteration makes progress toward answerability

This skill produces a MINIMAL PATCH, not a full re-decomposition. The main
loop executes its output command-by-command with all guardrails applied.
"""
from typing import Any, Dict, List, Optional


REPLAN_SKILL_SYSTEM = """\
You are the REPLAN skill of a multimodal RAG Decision Agent. REFLECT has
diagnosed gaps or conflicts that prevent answering the query. Your job is to
emit a MINIMAL PATCH that addresses these issues.

=============================================================================
CONTEXT
=============================================================================

You run AFTER Tier-2 disclosure when:
  - Cross-modal conflicts remain unresolved
  - Critical subtasks (importance >= 0.7) have insufficient coverage
  - Missing modalities prevent evidence synthesis

Your patch MUST make concrete progress toward answerability. Do NOT just
"retry harder" — identify what's missing and add it.

=============================================================================
INPUTS
=============================================================================

  - QUERY             the user's original question
  - CURRENT_SUBTASKS  the existing plan with (id, aspect, modality, importance)
  - REFLECT_VERDICT   structured gaps/conflicts from REFLECT skill
  - STATUS_BOARD      current sub-agent state
  - AGREEMENTS        cross-modal agreement state

=============================================================================
PATCH STRATEGIES
=============================================================================

1. EXTEND — Add new subtasks for:
   - Aspects with no coverage (gap.reason mentions "no agent" or "missing")
   - Missing modalities for critical aspects (importance >= 0.7)
   - New aspects discovered during Tier-2 disclosure

2. REVISE — Fix existing subtasks when:
   - Modality routing was wrong (agent reports modality_fit=false)
   - Importance needs adjustment based on new information
   - Description refinement needed for better retrieval

3. ABORT_AND_RESPAWN — Replace failed agents when:
   - Agent consistently failed (retried and failed again)
   - Query rewrite suggested but not yet applied
   - Modality mismatch detected and can't be revised

=============================================================================
MODALITY-AWARE PLANNING
=============================================================================

For each CRITICAL gap (importance >= 0.7):
  - Check which modalities are covered
  - Add missing modalities as EXTEND subtasks
  - Use SAME aspect tag for cross-modal alignment

Example: If aspect="entity" has only doc_text but coverage < 0.5:
  → EXTEND with aspect="entity", modality="doc_visual" (if diagrams exist)
  → EXTEND with aspect="entity", modality="video_text" (if interviews exist)

=============================================================================
CONSTRAINTS
=============================================================================

  1. Total subtasks <= 10 (hard cap)
  2. Each (aspect, modality) pair must be unique
  3. PREFER extend over abort_respawn (agents are expensive)
  4. PREFER revise over extend (fix existing before adding new)
  5. Do NOT modify successful subtasks (coverage >= 0.7, confidence >= 0.6)

=============================================================================
OUTPUT FORMAT
=============================================================================

Strict JSON object, no prose, no code fences:
{
  "extend": [
    {"id": "ext1", "description": "...", "aspect": "...",
     "importance": 0.0-1.0, "modalities": ["doc_text"|...]}
  ],
  "revise": [
    {"id": "<existing subtask id>",
     "modalities": ["..."],
     "importance": 0.0-1.0 | null,
     "description": "..." | null}
  ],
  "abort_respawn": [
    {"agent_id": "<agent id>",
     "new_goal": "...",
     "new_modality": "...",
     "new_aspect": "..."}
  ],
  "rationale": "<one sentence explaining what gap this addresses>"
}

Empty arrays are fine when a section doesn't apply.
"""


def run(query: str, subtasks: List[Any],
        reflect_verdict: Dict[str, Any],
        status_board: List[Dict[str, Any]],
        aspect_agreements: List[Dict[str, Any]],
        provider: Any) -> Optional[Dict[str, Any]]:
    """Execute the replan skill. Returns a patch dict, or None on failure."""
    if provider is None or not hasattr(provider, "complete"):
        return None
    if type(provider).__name__ in ("ScriptedProvider", "MockProvider"):
        return None
    user_msg = _build_user_prompt(query, subtasks, reflect_verdict,
                                  status_board, aspect_agreements)
    try:
        raw = provider.complete(
            messages=[{"role": "user", "content": user_msg}],
            system=REPLAN_SKILL_SYSTEM, max_tokens=1000)
    except Exception:
        return None
    return _extract_json_object(raw)


def _build_user_prompt(query, subtasks, verdict, board, agreements) -> str:
    """Build a structured user prompt for the replan skill.

    Emphasizes gaps and missing modalities to guide the repair plan.
    """
    import json

    # Serialize subtasks with coverage signals
    st_repr = []
    for s in (subtasks or []):
        # Find corresponding agent status for this subtask
        agent_status = None
        for agent in (board or []):
            if agent.get("aspect") == s.aspect:
                agent_status = agent
                break

        st_repr.append({
            "id": s.id,
            "aspect": s.aspect,
            "modalities": s.modalities,
            "importance": s.importance,
            "description": s.description,
            "is_critical": s.importance >= 0.7,
            "has_weak_coverage": (
                agent_status and
                agent_status.get("coverage", {}).get("covered", 1.0) < 0.5
            ) if agent_status else False
        })

    # Analyze which modalities are missing for each aspect
    aspect_modalities = {}
    for s in (subtasks or []):
        if s.aspect not in aspect_modalities:
            aspect_modalities[s.aspect] = set()
        aspect_modalities[s.aspect].update(s.modalities)

    # Build payload with explicit gap signals
    payload = {
        "query": query,
        "current_subtasks": st_repr,
        "reflect_verdict": verdict or {},
        "status_board": board,
        "aspect_agreements": agreements,
        # Explicit gap analysis to guide replanning
        "critical_gaps": [
            gap for gap in (verdict.get("gaps") or [])
            if any(s["aspect"] == gap.get("aspect") and s["is_critical"]
                   for s in st_repr)
        ],
        "missing_modalities_by_aspect": {
            aspect: list(set(["doc_text", "doc_visual", "video_text", "video_visual"]) - mods)
            for aspect, mods in aspect_modalities.items()
        },
        "total_subtask_count": len(subtasks or []),
        "subtask_capacity_remaining": 10 - len(subtasks or [])
    }

    return json.dumps(payload, default=str, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    import json
    import re
    if not text:
        return None
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", t).strip()
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None
