"""Value-of-Information (VoI) gating for progressive disclosure.

Role in the architecture
------------------------
The Decision Agent decides WHAT it wants (summary / sketch / full evidence);
this module decides WHETHER that request is economically justified. It sits
between DA commands (REQUEST_EVIDENCE_SKETCH, REQUEST_FULL_EVIDENCE) and
pool authorization. VoI returns a structured decision:

    {allow: bool, voi: float, reason: str, components: {...}}

The Orchestrator records every decision into state.voi_decisions for ablation
analysis ("how often would VoI have blocked a request?").

Gating policy (approval behavior = option A: soft-block with retry)
-------------------------------------------------------------------
  - First request:   if VoI < threshold, DENY and record.
  - Retry request:   if already denied once AND DA issues again, APPROVE.
                     This lets DA override VoI when it has strong reasoning
                     the gating heuristic missed, without creating a free
                     pass to ignore VoI.
  - Hard rules bypass threshold: cross-agent conflict, high-importance gap,
    or mid-confidence uncertainty zone → always approve.

Budget handling (option B: degrade gracefully)
----------------------------------------------
  - If budget.used_tokens > budget.max_tokens, SKETCH/FULL are denied at
    the budget check BEFORE VoI. Run continues on Tier-2 only. No hard kill.

This file contains ONLY pure functions + the structured decision dataclass.
No provider/pool/orchestrator imports — so unit tests are cheap and it
stays replaceable for RL training later (state → action → reward).
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# ---- Tunable parameters (paper ablation dimensions) --------------------
# Threshold above which VoI approves a request.
DEFAULT_THRESHOLD = 0.3
# Weight on cost in the VoI formula: voi = value - lambda * cost.
DEFAULT_LAMBDA_COST = 0.5
# Estimated token cost of a tier-3 full-evidence request. Rough; used as a
# unitless cost signal (0-1 after budget normalization).
DEFAULT_FULL_TOKEN_ESTIMATE = 800
DEFAULT_SKETCH_TOKEN_ESTIMATE = 250
# Hard-rule: mid-confidence uncertainty zone. If any confidence axis is
# inside [LO, HI], treat it as "worth inspecting".
MID_CONF_LO, MID_CONF_HI = 0.5, 0.8
# Hard-rule: cross-agent conflict. Confidence variance across same-aspect
# agents exceeding this → conflict detected.
CONFLICT_CONF_GAP = 0.4
# Hard-rule: important subtask with inadequate coverage.
IMPORTANCE_HARD_THRESH = 0.8
COVERAGE_HARD_THRESH = 0.5


@dataclass
class GatingDecision:
    """Structured VoI decision. Always recorded in state.voi_decisions."""
    allow: bool
    voi: float                      # final VoI value (may be NaN if hard-ruled)
    reason: str                     # human-readable explanation
    components: Dict[str, float] = field(default_factory=dict)
    # Metadata
    agent_id: str = ""
    stage: str = ""                 # "sketch" | "full"
    retry_count: int = 0

    def to_trace(self) -> Dict[str, Any]:
        return asdict(self)


# ---- Core formulas ------------------------------------------------------

def _max_confidence(snap_row: Dict[str, Any]) -> float:
    """Max across three confidence axes. Unreported axes treated as 0."""
    conf = snap_row.get("confidence") or {}
    vals = [conf.get("retrieval_score"), conf.get("evidence_agreement"),
            conf.get("coverage")]
    return max((v for v in vals if v is not None), default=0.0)


def _coverage_covered(snap_row: Dict[str, Any]) -> float:
    cov = snap_row.get("coverage") or {}
    return float(cov.get("covered", 0.0))


def compute_uncertainty(snap_row: Dict[str, Any]) -> float:
    """Weighted blend of coverage-gap + ambiguity + missing-aspects signal.

    Aligned to spec 3.1: 0.4*(1-coverage) + 0.4*ambiguity + 0.2*missing_score.
    All three are [0,1]. Output is clamped to [0,1].
    """
    coverage = _coverage_covered(snap_row)
    ambiguity = snap_row.get("ambiguity")
    if ambiguity is None:
        ambiguity = 0.5  # agnostic default when pool couldn't compute
    missing = snap_row.get("missing_aspects") or []
    missing_score = min(len(missing) * 0.2, 1.0)

    u = (0.4 * (1.0 - coverage) +
         0.4 * float(ambiguity) +
         0.2 * missing_score)
    return max(0.0, min(1.0, u))


def compute_value(snap_row: Dict[str, Any], uncertainty: float) -> float:
    """Value of acquiring more evidence: high when confidence is low OR
    uncertainty is high. Spec 3.1: 0.5*(1-conf) + 0.5*uncertainty."""
    conf = _max_confidence(snap_row)
    v = 0.5 * (1.0 - conf) + 0.5 * float(uncertainty)
    return max(0.0, min(1.0, v))


def estimate_cost(budget, stage: str) -> float:
    """Unitless cost estimate = fraction of remaining budget the request
    would consume. Clamp to [0,1]."""
    per_req = (DEFAULT_FULL_TOKEN_ESTIMATE if stage == "full"
               else DEFAULT_SKETCH_TOKEN_ESTIMATE)
    if budget is None or budget.max_tokens <= 0:
        return 0.0
    return min(per_req / max(budget.max_tokens, 1), 1.0)


def detect_conflict(status_board: List[Dict[str, Any]]) -> bool:
    """Conflict if two+ agents on the same aspect disagree in confidence by
    more than CONFLICT_CONF_GAP."""
    by_aspect: Dict[str, List[float]] = {}
    for row in status_board:
        aspect = row.get("aspect") or row.get("modality")
        if not aspect:
            continue
        c = _max_confidence(row)
        by_aspect.setdefault(aspect, []).append(c)
    for confs in by_aspect.values():
        if len(confs) >= 2 and (max(confs) - min(confs)) > CONFLICT_CONF_GAP:
            return True
    return False


def budget_exceeded(budget) -> bool:
    if budget is None:
        return False
    return budget.used_tokens > budget.max_tokens


# ---- Gating decisions ---------------------------------------------------

def gate_request(
    snap_row: Dict[str, Any],
    status_board: List[Dict[str, Any]],
    subtask: Optional[Any],            # runtime.Subtask or None
    budget: Optional[Any],             # runtime.Budget or None
    stage: str,                        # "sketch" | "full" | "curated_*"
    retry_count: int = 0,
    threshold: float = DEFAULT_THRESHOLD,
    lambda_cost: float = DEFAULT_LAMBDA_COST,
    info_gain_saturated: bool = False,   # Round 4 — global saturation flag
) -> GatingDecision:
    """Central decision routine. Returns a GatingDecision.

    Order of checks:
      1. Budget exceeded → DENY (terminal, no retry override).
      2. Info-gain saturated AND stage is a "keep going" request → DENY
         (terminal for this stage; DA must STOP or switch to inspect).
         Retry override does NOT bypass saturation — that would defeat it.
      3. Retry override → APPROVE (DA has asked twice, we trust it).
      4. Hard rules → APPROVE (conflict / important-gap / mid-conf zone).
      5. VoI formula → compare (value - lambda*cost) to threshold.
    """
    agent_id = snap_row.get("agent_id", "")

    # 1. Budget check.
    if budget_exceeded(budget):
        return GatingDecision(
            allow=False, voi=float("nan"),
            reason="budget_exceeded",
            components={"budget_used": float(budget.used_tokens),
                        "budget_max": float(budget.max_tokens)},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    # 2. Saturation check — Round 4. When the run has stalled (no info
    # gain over the rolling window), DENY further retrieval-intensive
    # stages. The DA sees the denial reason and should switch to STOP
    # (or, rarely, a targeted INSPECT). Retry override is DELIBERATELY
    # bypassed here — the whole point is a hard stop.
    SATURATION_BLOCKED_STAGES = {
        "sketch", "full", "curated_light", "curated_raw"}
    if info_gain_saturated and stage in SATURATION_BLOCKED_STAGES:
        return GatingDecision(
            allow=False, voi=float("nan"),
            reason="info_gain_saturated",
            components={"stage": 0.0},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    # 3. Retry override (one-retry soft-block).
    if retry_count >= 1:
        return GatingDecision(
            allow=True, voi=float("nan"),
            reason="retry_override",
            components={},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    # 3. Hard rules.
    conflict = detect_conflict(status_board)
    if conflict:
        return GatingDecision(
            allow=True, voi=float("nan"),
            reason="hard_rule_conflict",
            components={"conflict": 1.0},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    covered = _coverage_covered(snap_row)
    if (subtask is not None and
            getattr(subtask, "importance", 0.0) > IMPORTANCE_HARD_THRESH and
            covered < COVERAGE_HARD_THRESH):
        return GatingDecision(
            allow=True, voi=float("nan"),
            reason="hard_rule_important_gap",
            components={"importance": float(subtask.importance),
                        "covered": covered},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    max_conf = _max_confidence(snap_row)
    if MID_CONF_LO <= max_conf <= MID_CONF_HI:
        return GatingDecision(
            allow=True, voi=float("nan"),
            reason="hard_rule_mid_conf",
            components={"max_conf": max_conf},
            agent_id=agent_id, stage=stage, retry_count=retry_count)

    # 4. VoI formula.
    uncertainty = compute_uncertainty(snap_row)
    value = compute_value(snap_row, uncertainty)
    cost = estimate_cost(budget, stage)
    voi = value - lambda_cost * cost

    return GatingDecision(
        allow=(voi > threshold),
        voi=voi,
        reason=("voi_pass" if voi > threshold else "voi_below_threshold"),
        components={
            "uncertainty": uncertainty,
            "value": value,
            "cost": cost,
            "max_conf": max_conf,
            "threshold": threshold,
            "lambda_cost": lambda_cost,
        },
        agent_id=agent_id, stage=stage, retry_count=retry_count)


def select_evidence_tier(snap_row: Dict[str, Any]) -> str:
    """Recommendation only — DA is still the decider. Returns the TIER the
    snapshot's uncertainty profile would suggest. Used in status board
    rendering so DA sees a recommendation alongside raw numbers.

    Thresholds (spec 4):  u<0.4 → TIER_2, u<0.7 → TIER_2_5, else TIER_3.
    """
    u = compute_uncertainty(snap_row)
    if u < 0.4:
        return "TIER_2"
    if u < 0.7:
        return "TIER_2_5"
    return "TIER_3"