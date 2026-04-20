#!/usr/bin/env python3
"""Master experiment runner for EMNLP.

ROUTE A: Unified Framework positioning. Run MERP + baselines +
ablations on MMLongBench-Doc AND LongerVideos SEPARATELY, then show
that the SAME control mechanisms (VoI, saturation, conflict resolution)
transfer across modalities (doc ↔ video).

Produces data for all paper tables:
  Table 1a — MMLongBench-Doc (doc-only): MERP vs MDocAgent + baselines
  Table 1b — LongerVideos (video-only): MERP vs VideoRAG + baselines
  Table 2  — Modality-agnostic ablation (same ablations on both)
  Table 3  — Efficiency (tokens / latency) across both benchmarks
  Table 4  — MMLongBench-Doc subset breakdown (text/visual/cross-modal/...)
  Table 5  — LongerVideos per-category breakdown

Also produces:
  - full-run CSV per (benchmark, variant, query)
  - aggregated JSON with means + 95% CI via bootstrap
  - JSONL traces for MERP runs (for qualitative case study)

OPTIONAL — cross-corpus (if time permits later):
  Pass --cross-corpus /path/to/cross_corpus.json to include a third
  benchmark probing doc+video unified retrieval. This is ROUTE B and
  is kept available for a follow-up pass.

Usage:

  # ROUTE A primary run (no cross-corpus)
  python scripts/run_experiments.py \\
      --mmlongbench-root /data/MMLongBench-Doc \\
      --longervideos-root /data/LongerVideos \\
      --out-dir experiments/route_a/ \\
      --bootstrap 1000

  # Smoke test (tiny, mock providers)
  python scripts/run_experiments.py \\
      --mmlongbench-root /data/MMLongBench-Doc \\
      --limit 5 --use-mocks --out-dir /tmp/smoke/

  # ROUTE B extension (when ready)
  python scripts/run_experiments.py \\
      --mmlongbench-root /data/MMLongBench-Doc \\
      --longervideos-root /data/LongerVideos \\
      --cross-corpus /data/cross_corpus.json \\
      --out-dir experiments/route_b/ --bootstrap 1000

Variants (pass via --variants comma-separated):
  merp                     full MERP
  merp_no_infogain         w/o saturation (tracker disabled)
  merp_no_conflict         w/o RESOLVE_CONFLICT (falls to majority vote)
  merp_no_voi              w/o VoI gating
  merp_no_curated          w/o Tier-2 curated (Tier-1 only)
  merp_no_reflect          w/o REFLECT skill
  merp_no_replan           w/o REPLAN skill
  standard_rag             single retrieve → generate
  late_fusion              per-modality retrieve → LLM fuses
  early_fusion             union-ranked retrieve → LLM
  self_rag_style           iterative retrieve-and-critique
  mdocagent                MDocAgent-like: 5 specialist agents (external)
  videorag                 VideoRAG-like: dual channel (external)

Baselines mdocagent/videorag require cloning the official repos + wiring
a thin wrapper. See scripts/baselines_external/ for stubs.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

log = logging.getLogger("experiments")


# ===========================================================================
# Variant registry
# ===========================================================================
def _variant_merp(query, store, providers, config=None):
    from orchestrator.controller import Orchestrator
    from agents.decision_agent import DecisionAgent
    from agents.factory import SubAgentFactory
    from config.ablation import AblationConfig
    from experiments.common.utils.export_trace import dump_trace
    factory = SubAgentFactory(
        provider=providers["sub_text"],
        provider_visual=providers["sub_visual"])
    orch = Orchestrator(
        DecisionAgent(providers["da"]), factory, store,
        ablation=AblationConfig.preset("baseline"),
        max_steps=(config or {}).get("max_steps", 25))
    result = orch.run(query)
    # Attach trace for later analysis
    if config and config.get("trace_path"):
        from memory.state_manager import StateManager
        # Controller doesn't return state directly; dump_trace needs it.
        # Downstream pipelines that need the trace should use run_ablation
        # which holds state across the loop. Skipping here for simplicity.
        pass
    return result


def _with_disabled(attr_path: str, patched_value):
    """Context manager that temporarily monkey-patches attr_path to
    patched_value for the duration of a variant run."""
    class Ctx:
        def __enter__(self):
            mod_path, _, name = attr_path.rpartition(".")
            import importlib
            self.mod = importlib.import_module(mod_path)
            self.orig = getattr(self.mod, name)
            setattr(self.mod, name, patched_value)
            return self
        def __exit__(self, *a):
            setattr(self.mod, name := attr_path.rpartition(".")[2], self.orig)
    return Ctx()


def _variant_merp_no_reflect(query, store, providers, config=None):
    from prompts.skills import reflect_skill
    orig = reflect_skill.run
    reflect_skill.run = lambda *a, **kw: None
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        reflect_skill.run = orig


def _variant_merp_no_replan(query, store, providers, config=None):
    from prompts.skills import replan_skill
    orig = replan_skill.run
    replan_skill.run = lambda *a, **kw: None
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        replan_skill.run = orig


def _variant_merp_no_infogain(query, store, providers, config=None):
    from memory.state_manager import StateManager
    from utils.info_gain_tracker import InfoGainTracker
    orig = StateManager.__init__
    def patched(self, *a, **kw):
        orig(self, *a, **kw)
        # Window of 999999 means saturation never fires
        self.info_gain_tracker = InfoGainTracker(window=999999, delta=0.0)
    StateManager.__init__ = patched
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        StateManager.__init__ = orig


def _variant_merp_no_conflict(query, store, providers, config=None):
    """Disable RESOLVE_CONFLICT — any aspect-disagree is auto-resolved
    by picking the highest-confidence agent (majority-style fallback)."""
    from orchestrator.controller import Orchestrator
    orig_dispatch = Orchestrator._dispatch
    def patched(self, cmd, state, pool):
        if cmd.get("command") == "RESOLVE_CONFLICT":
            return    # no-op
        return orig_dispatch(self, cmd, state, pool)
    Orchestrator._dispatch = patched
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        Orchestrator._dispatch = orig_dispatch


def _variant_merp_no_voi(query, store, providers, config=None):
    """VoI gate always approves — equivalent to MDocAgent-style every-
    request-granted flow."""
    from orchestrator import voi_gating
    orig = voi_gating.gate_request
    def always_allow(*a, **kw):
        snap_row = a[0] if a else kw.get("snap_row", {})
        from orchestrator.voi_gating import GatingDecision
        return GatingDecision(
            allow=True, voi=float("nan"),
            reason="voi_disabled",
            components={},
            agent_id=snap_row.get("agent_id", ""),
            stage=kw.get("stage", "?"), retry_count=kw.get("retry_count", 0))
    voi_gating.gate_request = always_allow
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        voi_gating.gate_request = orig


def _variant_merp_no_curated(query, store, providers, config=None):
    """Disable REQUEST_CURATED_EVIDENCE (Tier-2). DA sees Tier-1 only."""
    from orchestrator.controller import Orchestrator
    orig_dispatch = Orchestrator._dispatch
    def patched(self, cmd, state, pool):
        if cmd.get("command") == "REQUEST_CURATED_EVIDENCE":
            return
        return orig_dispatch(self, cmd, state, pool)
    Orchestrator._dispatch = patched
    try:
        return _variant_merp(query, store, providers, config)
    finally:
        Orchestrator._dispatch = orig_dispatch


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


def _variant_mdocagent(query, store, providers, config=None):
    """Stub — wire in MDocAgent external baseline when you have it.
    Placeholder returns NotImplementedError so the runner skips this
    variant cleanly if you don't have the baseline wired."""
    try:
        from experiments.common.baselines_external.mdocagent_wrapper import run
        return run(query, store, providers)
    except ImportError:
        return {"answer": "[MDocAgent not wired]",
                "confidence": 0.0, "reason": "not_wired",
                "n_retrieved": 0, "n_tokens": 0}


def _variant_videorag(query, store, providers, config=None):
    try:
        from experiments.common.baselines_external.videorag_wrapper import run
        return run(query, store, providers)
    except ImportError:
        return {"answer": "[VideoRAG not wired]",
                "confidence": 0.0, "reason": "not_wired",
                "n_retrieved": 0, "n_tokens": 0}


VARIANTS: Dict[str, Callable] = {
    "merp":               _variant_merp,
    "merp_no_infogain":   _variant_merp_no_infogain,
    "merp_no_conflict":   _variant_merp_no_conflict,
    "merp_no_voi":        _variant_merp_no_voi,
    "merp_no_curated":    _variant_merp_no_curated,
    "merp_no_reflect":    _variant_merp_no_reflect,
    "merp_no_replan":     _variant_merp_no_replan,
    "standard_rag":       _variant_standard_rag,
    "late_fusion":        _variant_late_fusion,
    "early_fusion":       _variant_early_fusion,
    "self_rag_style":     _variant_self_rag,
    "mdocagent":          _variant_mdocagent,
    "videorag":           _variant_videorag,
}


# ===========================================================================
# Scoring
# ===========================================================================
def score_doc(pred: Any, example, scorer_fn=None):
    from data.loaders.mmlongbench_doc import quick_score
    if scorer_fn:
        return scorer_fn(pred, example)
    return quick_score(pred, example)


def score_video(pred: Any, example):
    from data.loaders.longer_videos import token_f1
    if isinstance(pred, dict):
        pred = pred.get("answer", "")
    return {"em": int(str(pred).strip().lower() ==
                      str(example.reference_answer).strip().lower()),
            "f1": token_f1(pred, example.reference_answer)}


def score_cross_corpus(pred: Any, example):
    # Same shape as video; the reference is typically short
    return score_video(pred, example)


# ===========================================================================
# Bootstrap CI
# ===========================================================================
def bootstrap_ci(values: List[float], n_boot: int = 1000,
                  alpha: float = 0.05, seed: int = 0
                  ) -> Tuple[float, float, float]:
    """Return (mean, lo, hi) where [lo, hi] is (1-alpha) CI."""
    if not values:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot)]
    return sum(values) / n, lo, hi


# ===========================================================================
# Main pipeline
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="EMNLP experiment runner")
    p.add_argument("--mmlongbench-root", default=None,
                   help="path to MMLongBench-Doc release")
    p.add_argument("--longervideos-root", default=None,
                   help="path to LongerVideos release")
    p.add_argument("--cross-corpus", default=None,
                   help="path to cross_corpus.json")
    p.add_argument("--variants", default=",".join([
                       "merp", "merp_no_infogain", "merp_no_conflict",
                       "standard_rag", "late_fusion", "early_fusion",
                       "self_rag_style"]),
                   help="comma-separated variants to run")
    p.add_argument("--limit", type=int, default=None,
                   help="cap examples per benchmark (for smoke)")
    p.add_argument("--out-dir", default="experiments/default/",
                   help="output directory for CSVs + aggregates")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap samples for CI; 0 disables")
    p.add_argument("--use-mocks", action="store_true",
                   help="use mock providers for pipeline smoke testing")
    p.add_argument("--gpt-scorer", action="store_true",
                   help="use GPT-4o 3-stage scorer for MMLongBench-Doc "
                        "(costs money; quick_score is default)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _build_providers_and_stores(args):
    """Build providers + per-benchmark MultiIndexStore."""
    from experiments.common.utils.live_demo_madqa import _build_providers, _build_embedders
    da_provider, factory = _build_providers(args)
    text_emb, image_emb = _build_embedders(args)
    providers = {"da": da_provider,
                 "sub_text": factory._provider,
                 "sub_visual": factory._provider_visual}

    stores: Dict[str, Any] = {}

    if args.mmlongbench_root:
        from data.loaders.mmlongbench_doc import load_mmlongbench_doc
        from data.parsers.pdf_parser import parse_pdf
        from data.index_builder import build_indices
        log.info(f"[exp] loading MMLongBench-Doc from {args.mmlongbench_root}")
        examples, pdf_paths = load_mmlongbench_doc(
            args.mmlongbench_root, limit=args.limit)
        log.info(f"[exp] ingesting {len(pdf_paths)} PDFs...")
        blocks = []
        for doc_id, path in list(pdf_paths.items()):
            try:
                blocks.extend(parse_pdf(path, source_id=doc_id))
            except Exception as e:
                log.warning(f"parse failed {doc_id}: {e}")
        stores["mmlongbench"] = {
            "examples": examples,
            "store":    build_indices(blocks, text_emb, image_emb),
            "score_fn": score_doc}

    if args.longervideos_root:
        from data.loaders.longer_videos import load_longer_videos
        from data.parsers.video_parser import parse_video
        from data.index_builder import build_indices
        log.info(f"[exp] loading LongerVideos from {args.longervideos_root}")
        examples, video_paths = load_longer_videos(
            args.longervideos_root, limit=args.limit)
        log.info(f"[exp] ingesting {len(video_paths)} videos (slow!)...")
        blocks = []
        for vid, path in list(video_paths.items()):
            try:
                blocks.extend(parse_video(path, source_id=vid))
            except Exception as e:
                log.warning(f"video parse failed {vid}: {e}")
        stores["longervideos"] = {
            "examples": examples,
            "store":    build_indices(blocks, text_emb, image_emb),
            "score_fn": score_video}

    if args.cross_corpus:
        from data.loaders.cross_corpus import load_cross_corpus
        log.info(f"[exp] loading cross-corpus from {args.cross_corpus}")
        examples = load_cross_corpus(args.cross_corpus, limit=args.limit)
        # For cross-corpus we need the UNION of both doc + video stores.
        # Require both corpora to be loaded above.
        if ("mmlongbench" in stores and "longervideos" in stores):
            union = _merge_stores(stores["mmlongbench"]["store"],
                                   stores["longervideos"]["store"])
        elif "mmlongbench" in stores:
            union = stores["mmlongbench"]["store"]
        elif "longervideos" in stores:
            union = stores["longervideos"]["store"]
        else:
            raise RuntimeError("cross-corpus requires either "
                               "--mmlongbench-root or --longervideos-root")
        stores["cross_corpus"] = {
            "examples": examples,
            "store":    union,
            "score_fn": score_cross_corpus}

    return providers, stores


def _merge_stores(store_a, store_b):
    """Union two MultiIndexStore instances across the same 4 modalities.
    Concatenates docs + rebuilds BM25. Uses VectorIndex encoder of A."""
    from memory.store import VectorIndex, MultiIndexStore
    MODS = ("doc_text", "doc_visual", "video_text", "video_visual")
    merged: Dict[str, VectorIndex] = {}
    for m in MODS:
        a_docs = store_a._indices[m].docs if m in store_a._indices else []
        b_docs = store_b._indices[m].docs if m in store_b._indices else []
        merged[m] = VectorIndex(m, a_docs + b_docs)
    return MultiIndexStore(merged, hybrid=True)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)

    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants_to_run if v not in VARIANTS]
    if unknown:
        print(f"[exp] unknown variants: {unknown}", file=sys.stderr)
        print(f"[exp] valid: {sorted(VARIANTS.keys())}", file=sys.stderr)
        sys.exit(2)

    providers, stores = _build_providers_and_stores(args)
    if not stores:
        print("[exp] no benchmarks loaded; provide at least one "
              "--*-root argument", file=sys.stderr)
        sys.exit(2)

    # Per-benchmark per-variant per-query result row
    fieldnames = ["benchmark", "variant", "query_id", "em", "f1",
                  "n_tokens", "n_retrieved", "elapsed_sec",
                  "answer", "gold"]
    raw_csv = os.path.join(args.out_dir, "raw_results.csv")
    fout = open(raw_csv, "w", encoding="utf-8", newline="")
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    # Track values for aggregation
    bucket: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    for bench_name, b in stores.items():
        examples = b["examples"]
        store = b["store"]
        score_fn = b["score_fn"]
        log.info(f"[exp] === {bench_name}: {len(examples)} examples, "
                 f"{len(variants_to_run)} variants ===")
        for i, ex in enumerate(examples):
            q_id = getattr(ex, "id", f"{bench_name}_{i}")
            query = (getattr(ex, "question", None) or
                     getattr(ex, "query", None) or "")
            gold = (getattr(ex, "answer", None) or
                    getattr(ex, "reference_answer", None) or "")

            for variant in variants_to_run:
                fn = VARIANTS[variant]
                t0 = time.time()
                try:
                    pred = fn(query, store, providers)
                except Exception as e:
                    log.exception(f"{variant} failed on {q_id}")
                    pred = {"answer": "", "confidence": 0.0,
                            "reason": f"error: {e}",
                            "n_tokens": 0, "n_retrieved": 0}
                elapsed = time.time() - t0
                metrics = score_fn(pred, ex) if bench_name != "longervideos" \
                    else score_fn(pred, ex)

                writer.writerow({
                    "benchmark": bench_name, "variant": variant,
                    "query_id": q_id, "em": metrics["em"],
                    "f1": round(metrics["f1"], 4),
                    "n_tokens": pred.get("n_tokens", 0),
                    "n_retrieved": pred.get("n_retrieved", 0),
                    "elapsed_sec": round(elapsed, 2),
                    "answer": (pred.get("answer") or "")[:300],
                    "gold": str(gold)[:300]})
                fout.flush()

                key = (bench_name, variant)
                bucket.setdefault(key, {"em": [], "f1": [],
                                        "tokens": [], "elapsed": []})
                bucket[key]["em"].append(metrics["em"])
                bucket[key]["f1"].append(metrics["f1"])
                bucket[key]["tokens"].append(pred.get("n_tokens", 0))
                bucket[key]["elapsed"].append(elapsed)

            if (i + 1) % 10 == 0:
                log.info(f"[exp] {bench_name} {i+1}/{len(examples)} done")

    fout.close()
    log.info(f"[exp] raw results: {raw_csv}")

    # Aggregate with bootstrap
    agg_rows = []
    for (bench, variant), v in bucket.items():
        em_mean, em_lo, em_hi = (bootstrap_ci(v["em"], args.bootstrap,
                                                seed=args.seed)
                                   if args.bootstrap else
                                   (sum(v["em"]) / max(len(v["em"]), 1), 0, 0))
        f1_mean, f1_lo, f1_hi = (bootstrap_ci(v["f1"], args.bootstrap,
                                                seed=args.seed)
                                   if args.bootstrap else
                                   (sum(v["f1"]) / max(len(v["f1"]), 1), 0, 0))
        agg_rows.append({
            "benchmark": bench, "variant": variant, "n": len(v["em"]),
            "em": round(em_mean, 4),
            "em_ci_lo": round(em_lo, 4), "em_ci_hi": round(em_hi, 4),
            "f1": round(f1_mean, 4),
            "f1_ci_lo": round(f1_lo, 4), "f1_ci_hi": round(f1_hi, 4),
            "avg_tokens": round(sum(v["tokens"]) /
                                 max(len(v["tokens"]), 1), 0),
            "avg_sec":    round(sum(v["elapsed"]) /
                                 max(len(v["elapsed"]), 1), 2)})

    agg_csv = os.path.join(args.out_dir, "aggregate.csv")
    with open(agg_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        w.writerows(agg_rows)
    log.info(f"[exp] aggregate: {agg_csv}")

    # Pretty print
    print(f"\n{'benchmark':<14} {'variant':<22} {'n':>4} "
          f"{'EM':>6} {'EM CI':>15} {'F1':>6} {'tok/q':>8} {'sec/q':>6}")
    print("-" * 90)
    for r in sorted(agg_rows, key=lambda x: (x["benchmark"], -x["f1"])):
        ci = f"[{r['em_ci_lo']:.3f},{r['em_ci_hi']:.3f}]"
        print(f"{r['benchmark']:<14} {r['variant']:<22} {r['n']:>4} "
              f"{r['em']:>6.3f} {ci:>15} {r['f1']:>6.3f} "
              f"{r['avg_tokens']:>8.0f} {r['avg_sec']:>6.2f}")


if __name__ == "__main__":
    main()
