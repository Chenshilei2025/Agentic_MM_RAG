#!/usr/bin/env python3
"""Trace exporter — dumps one MERP run's decision traces as JSONL.

Use this on each MADQA query to build the trace corpus you'll analyze in
the paper. Captures:
  - reflect_verdicts       every REFLECT call's output
  - replan_traces          every REPLAN's patch and what was applied
  - voi_decisions          every VoI gate decision (allow/deny + reason)
  - conflict_resolutions   every RESOLVE_CONFLICT decision
  - subtasks               final decomposition
  - budget.tier_breakdown  per-tier token accounting
  - info_gain_snapshot     saturation state at end of run
  - final answer + pool status board

Usage:
    # At end of Orchestrator.run(), call dump_trace(state, pool, answer)
    # Or as a CLI for post-hoc analysis:

    python scripts/export_trace.py \
        --run-output madqa_results.jsonl \
        --out-dir traces/
"""
from __future__ import annotations
import json
import os
from dataclasses import is_dataclass, asdict
from typing import Any, Dict


def dump_trace(state, pool, final_answer: Dict[str, Any],
               query: str = "",
               query_id: str = "",
               out_path: str = None) -> Dict[str, Any]:
    """Collect a full trace into one dict; optionally write it to JSONL.

    Args:
      state          StateManager after run
      pool           EvidencePool after run
      final_answer   {"answer", "confidence", "reason"} from Orchestrator
      query_id       MADQA example id, used as trace key
      out_path       optional; append to this JSONL file if given

    Returns:
      The trace dict.
    """
    trace = {
        "query_id": query_id or "",
        "query": query,
        "final": final_answer,
        # Subtask plan
        "subtasks": [_dc_to_dict(s) for s in (state.subtasks or [])],
        # Decision loop state
        "step_count": state.step_count,
        "reflect_verdicts": list(state.reflect_verdicts or []),
        "replan_traces": list(state.replan_traces or []),
        "voi_decisions": list(state.voi_decisions or []),
        "revise_trace": list(state.revise_trace or []),
        "conflict_resolutions": dict(state.conflict_resolutions or {}),
        "aborted_agents": sorted(list(state.aborted_agents or [])),
        # Budget / cost
        "budget": {
            "max_tokens": getattr(state.budget, "max_tokens", None)
                          if state.budget else None,
            "used_tokens": getattr(state.budget, "used_tokens", None)
                           if state.budget else None,
            "tier_breakdown": dict(getattr(state.budget, "tier_breakdown",
                                           {}) or {})
                              if state.budget else {},
        },
        # Saturation
        "info_gain_snapshot": (state.info_gain_tracker.snapshot()
                               if hasattr(state, "info_gain_tracker")
                               else None),
        # Final pool state
        "final_status_board": pool.status_board(include_terminated=True),
        "aspect_agreements": pool.aspect_agreements(),
        "max_confidence": pool.max_confidence(),
        "coverage": pool.coverage(),
    }
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, default=str, ensure_ascii=False) + "\n")
    return trace


def _dc_to_dict(obj):
    """Serialize a dataclass or pass through dict/primitive."""
    if is_dataclass(obj):
        d = asdict(obj)
        # drop heavy vectors from subtask export — keep just the dim
        if "embedding" in d and d["embedding"]:
            d["embedding"] = f"<len={len(d['embedding'])} vec>"
        return d
    return obj


# ---------------------------------------------------------------------------
# CLI — aggregate traces for paper analysis (not used during runs)
# ---------------------------------------------------------------------------
def _cli():
    import argparse
    p = argparse.ArgumentParser(
        description="Summarise a JSONL of MERP run traces into a paper-"
                    "friendly aggregate (counts of reflect/replan calls, "
                    "avg saturation turn, VoI denial rates, etc)")
    p.add_argument("--traces", required=True,
                   help="path to JSONL of trace dicts (from dump_trace)")
    p.add_argument("--out", default="trace_summary.json")
    args = p.parse_args()

    reflect_counts = []
    replan_counts = []
    voi_denial_rates = []
    saturation_turns = []
    tier_tokens: Dict[str, int] = {}

    with open(args.traces, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                t = json.loads(line)
            except json.JSONDecodeError:
                continue
            reflect_counts.append(len(t.get("reflect_verdicts") or []))
            replan_counts.append(len(t.get("replan_traces") or []))
            voi = t.get("voi_decisions") or []
            if voi:
                denials = sum(1 for d in voi if not d.get("allow"))
                voi_denial_rates.append(denials / len(voi))
            tb = (t.get("budget") or {}).get("tier_breakdown") or {}
            for k, v in tb.items():
                tier_tokens[k] = tier_tokens.get(k, 0) + v
            sat = (t.get("info_gain_snapshot") or {})
            if sat.get("saturated"):
                hist = sat.get("history") or []
                if hist:
                    saturation_turns.append(hist[0].get("turn", 0))

    def _avg(xs): return sum(xs) / len(xs) if xs else 0.0
    summary = {
        "n_traces": len(reflect_counts),
        "avg_reflect_calls_per_run": _avg(reflect_counts),
        "avg_replan_calls_per_run": _avg(replan_counts),
        "avg_voi_denial_rate": _avg(voi_denial_rates),
        "fraction_saturated_runs":
            sum(1 for x in saturation_turns) / max(len(reflect_counts), 1),
        "avg_saturation_turn": _avg(saturation_turns),
        "tier_token_totals": tier_tokens,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
