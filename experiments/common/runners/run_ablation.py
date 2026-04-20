#!/usr/bin/env python3
"""Ablation runner for the MADQA experiments.

Runs multiple VARIANTS over the same corpus and query set, collects:
  - EM, F1
  - n_tokens (total across DA + sub-agents + skills)
  - n_retrieved
  - n_iters / n_turns
  - per-tier token breakdown (MERP only)

and writes a comparison CSV.

Variants supported:
  - merp                 full MERP system (your method)
  - merp_no_reflect      MERP with reflect_skill disabled (heuristic fallback only)
  - merp_no_replan       MERP with REPLAN command disabled
  - merp_no_curated      MERP without Tier-2 curated disclosure
  - merp_no_info_gain    MERP without saturation check
  - standard_rag         baseline
  - late_fusion          baseline
  - early_fusion         baseline
  - self_rag_style       baseline

Usage:
    python scripts/run_ablation.py \
        --root /data/madqa \
        --limit 50 \
        --variants merp,late_fusion,self_rag_style \
        --out ablation_results.csv

All variants share the same MultiIndexStore built once up-front (so
retrieval quality is held constant).
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

# Let us import from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
@dataclass
class Variant:
    name: str
    run_fn: Callable[[str, Any, Any], Dict[str, Any]]
    description: str = ""


def _variant_merp(query, store, providers, config=None):
    """Full MERP system."""
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from config.ablation import AblationConfig

    factory = SubAgentFactory(
        provider=providers["sub_text"],
        provider_visual=providers["sub_visual"])
    orch = Orchestrator(
        DecisionAgent(providers["da"]), factory, store,
        ablation=AblationConfig.preset("baseline"),
        max_steps=config.get("max_steps", 25) if config else 25)
    result = orch.run(query)
    # Extract token breakdown if available
    # The orchestrator's last_state_trace may or may not expose budget;
    # best-effort via result enrichment would need an API change. For now
    # we report only used_tokens on the result.
    result.setdefault("n_tokens", 0)
    return result


def _variant_merp_no_reflect(query, store, providers, config=None):
    """MERP with reflect_skill disabled — DA falls through to heuristic."""
    # Monkey-patch reflect_skill.run to return None → heuristic fallback
    from prompts.skills import reflect_skill
    _orig = reflect_skill.run
    reflect_skill.run = lambda *a, **kw: None
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        reflect_skill.run = _orig


def _variant_merp_no_replan(query, store, providers, config=None):
    """MERP with REPLAN disabled — replan skill returns None."""
    from prompts.skills import replan_skill
    _orig = replan_skill.run
    replan_skill.run = lambda *a, **kw: None
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        replan_skill.run = _orig


def _variant_merp_no_info_gain(query, store, providers, config=None):
    """MERP with saturation check disabled (huge window → never saturates)."""
    from memory.state_manager import StateManager
    _orig = StateManager.__init__
    def patched(self, *a, **kw):
        _orig(self, *a, **kw)
        from utils.info_gain_tracker import InfoGainTracker
        self.info_gain_tracker = InfoGainTracker(window=99999, delta=0.0)
    StateManager.__init__ = patched
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        StateManager.__init__ = _orig


def _variant_standard_rag(query, store, providers, config=None):
    from baselines.standard_rag import run
    return run(query, store, providers["da"], k=5)


def _variant_late_fusion(query, store, providers, config=None):
    from baselines.late_fusion_rag import run
    return run(query, store, providers["da"], k_per_modality=3)


def _variant_early_fusion(query, store, providers, config=None):
    from baselines.early_fusion_rag import run
    return run(query, store, providers["da"], k=8)


def _variant_self_rag(query, store, providers, config=None):
    from baselines.self_rag_style import run
    return run(query, store, providers["da"], k=5, max_iters=3)


VARIANTS: Dict[str, Callable] = {
    "merp":                _variant_merp,
    "merp_no_reflect":     _variant_merp_no_reflect,
    "merp_no_replan":      _variant_merp_no_replan,
    "merp_no_info_gain":   _variant_merp_no_info_gain,
    "standard_rag":        _variant_standard_rag,
    "late_fusion":         _variant_late_fusion,
    "early_fusion":        _variant_early_fusion,
    "self_rag_style":      _variant_self_rag,
}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def score_example(pred: Any, gold: Any) -> Dict[str, float]:
    """EM + token-F1."""
    if isinstance(pred, dict):
        pred = pred.get("answer", "")
    gold_list = gold if isinstance(gold, list) else [gold]
    pred_str = str(pred or "").lower().strip()
    em = int(any(pred_str == str(g).lower().strip() for g in gold_list))
    def f1(pt, gt):
        if not pt or not gt: return 0.0
        common = set(pt) & set(gt)
        if not common: return 0.0
        p = len(common) / len(pt)
        r = len(common) / len(gt)
        return 2 * p * r / (p + r)
    pt = pred_str.split()
    best_f1 = max((f1(pt, str(g).lower().split()) for g in gold_list),
                  default=0.0)
    return {"em": em, "f1": best_f1}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="MADQA ablation runner")
    p.add_argument("--root", required=True, help="MADQA release root")
    p.add_argument("--split", default="dev")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--variants", default="merp,late_fusion,self_rag_style",
                   help="comma-separated subset of "
                        + ",".join(VARIANTS.keys()))
    p.add_argument("--out", default="ablation_results.csv")
    p.add_argument("--use-mocks", action="store_true",
                   help="use MockBGE + Mock CLIP + ScriptedProvider, for "
                        "pipeline smoke-testing without deps/GPU")
    return p.parse_args()


def _build_providers_and_store(args):
    # same helpers as live_demo
    from experiments.common.utils.live_demo_madqa import (_build_embedders, _build_providers,
                                                           _ingest_pdfs)
    from data.loaders.madqa import load_madqa

    examples, pdf_paths = load_madqa(args.root, split=args.split,
                                     limit=args.limit)
    text_emb, image_emb = _build_embedders(args)
    referenced = {e.doc_id for e in examples if e.doc_id in pdf_paths}
    store = _ingest_pdfs({d: pdf_paths[d] for d in referenced},
                         text_emb, image_emb)
    da_provider, factory = _build_providers(args)
    # We want per-provider knobs: the variants use these directly
    providers = {
        "da": da_provider,
        "sub_text": factory._provider,
        "sub_visual": factory._provider_visual,
    }
    return examples, store, providers


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants_to_run if v not in VARIANTS]
    if unknown:
        print(f"[ablation] unknown variants: {unknown}. Valid: "
              f"{sorted(VARIANTS.keys())}", file=sys.stderr)
        sys.exit(2)

    log.info(f"[ablation] loading dataset + building store...")
    examples, store, providers = _build_providers_and_store(args)
    log.info(f"[ablation] {len(examples)} examples loaded, "
             f"running variants: {variants_to_run}")

    # Results: one row per (variant, query)
    fieldnames = ["variant", "query_id", "em", "f1", "n_tokens",
                  "n_retrieved", "n_iters", "elapsed_sec",
                  "answer", "gold"]
    with open(args.out, "w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        agg: Dict[str, Dict[str, float]] = {
            v: {"em": 0.0, "f1": 0.0, "tokens": 0.0, "n": 0}
            for v in variants_to_run}

        for i, ex in enumerate(examples):
            log.info(f"[ablation] QA {i+1}/{len(examples)} id={ex.id}")
            for variant_name in variants_to_run:
                run_fn = VARIANTS[variant_name]
                t0 = time.time()
                try:
                    result = run_fn(ex.question, store, providers)
                except Exception as e:
                    log.exception(f"{variant_name} failed on {ex.id}")
                    result = {"answer": "", "confidence": 0,
                              "reason": f"error: {e}",
                              "n_tokens": 0, "n_retrieved": 0}
                elapsed = time.time() - t0
                metrics = score_example(result.get("answer", ""), ex.answer)
                writer.writerow({
                    "variant": variant_name,
                    "query_id": ex.id,
                    "em": metrics["em"],
                    "f1": round(metrics["f1"], 4),
                    "n_tokens": result.get("n_tokens", 0),
                    "n_retrieved": result.get("n_retrieved", 0),
                    "n_iters": result.get("n_iters", result.get("step_count", 0)),
                    "elapsed_sec": round(elapsed, 2),
                    "answer": (result.get("answer") or "")[:300],
                    "gold": str(ex.answer)[:300],
                })
                fout.flush()
                agg[variant_name]["em"] += metrics["em"]
                agg[variant_name]["f1"] += metrics["f1"]
                agg[variant_name]["tokens"] += result.get("n_tokens", 0)
                agg[variant_name]["n"] += 1

    # Print aggregate
    print("\n=== AGGREGATE ===")
    print(f"{'variant':<20} {'n':>4} {'EM':>6} {'F1':>6} {'avg_tokens':>10}")
    for v in variants_to_run:
        a = agg[v]
        n = max(a["n"], 1)
        print(f"{v:<20} {a['n']:>4} {a['em']/n:>6.3f} "
              f"{a['f1']/n:>6.3f} {a['tokens']/n:>10.0f}")
    print(f"\n[ablation] full results written to {args.out}")


if __name__ == "__main__":
    main()
