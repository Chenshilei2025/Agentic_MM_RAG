#!/usr/bin/env python
"""Benchmark runner. Drives (ablation × example) matrix, produces paper tables.

USAGE
  # Smoke test, scripted provider (no API key needed):
  python scripts/benchmark.py --dataset demo --n-examples 3 \
      --provider scripted --ablations baseline

  # Real experiment (needs ANTHROPIC_API_KEY):
  python scripts/benchmark.py --dataset demo --n-examples 10 \
      --provider anthropic \
      --ablations baseline,no_tier3_gate,no_rerank,no_parallelism,combined \
      --out-dir results/

OUTPUTS
  <out_dir>/runs/<ablation>/<example_id>.json  per-run trace
  <out_dir>/table1.csv                          main ablation table (CSV)
  <out_dir>/table1.tex                          paper-ready LaTeX
  <out_dir>/summary.json                        full nested summary
"""
import argparse, json, os, sys, time
from dataclasses import asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.datasets.demo_dataset import DemoDataset
from benchmarks.harness import run_one, dump_record
from benchmarks.aggregator import (summarize_runs, write_csv, write_latex,
                                   write_json)
from utils.logger import get_logger

log = get_logger("benchmark")

DATASETS = {"demo": DemoDataset}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="demo", choices=list(DATASETS.keys()))
    ap.add_argument("--n-examples", type=int, default=None,
                    help="limit number of examples (None = all)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--provider", default="scripted",
                    choices=["scripted", "anthropic"])
    ap.add_argument("--model", default=None, help="override ANTHROPIC_MODEL")
    ap.add_argument("--ablations", default="baseline",
                    help="comma-separated ablation preset names")
    ap.add_argument("--max-steps", type=int, default=15)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--step-timeout", type=float, default=60.0)
    ap.add_argument("--out-dir", default="results/")
    args = ap.parse_args()

    if args.provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        return 2

    ds = DATASETS[args.dataset]()
    examples = ds.load_examples(n=args.n_examples, seed=args.seed)
    ablations = [a.strip() for a in args.ablations.split(",") if a.strip()]

    log.info(f"[bench] dataset={args.dataset}  n={len(examples)}  "
             f"ablations={ablations}  provider={args.provider}")
    t0 = time.time()

    # Per-ablation run buckets
    summary: dict = {}

    for ab in ablations:
        log.info(f"\n{'='*60}\n  ABLATION: {ab}\n{'='*60}")
        records = []
        for ex in examples:
            log.info(f"[run] {ab}/{ex.id}: {ex.query[:80]}")
            # Rebuild the store per run so no state leaks between examples.
            store = ds.build_store()
            try:
                rec = run_one(ex, ab, store, args.provider,
                              model=args.model,
                              max_steps=args.max_steps,
                              max_workers=args.max_workers,
                              step_timeout=args.step_timeout)
            except Exception as e:
                log.error(f"[run-fail] {ab}/{ex.id}: {e}")
                continue
            dump_record(rec, os.path.join(args.out_dir, "runs", ab))
            records.append(asdict(rec))
            log.info(f"[done]  em={rec.em:.2f}  f1={rec.f1:.2f}  "
                     f"tokens={rec.input_tokens+rec.output_tokens}  "
                     f"wall={rec.wall_clock_s:.2f}s")
        summary[ab] = summarize_runs(records)

    # Emit artifacts
    write_csv(summary,   os.path.join(args.out_dir, "table1.csv"))
    write_latex(summary, os.path.join(args.out_dir, "table1.tex"))
    write_json(summary,  os.path.join(args.out_dir, "summary.json"))

    elapsed = time.time() - t0
    log.info(f"\n[bench] done in {elapsed:.1f}s")
    log.info(f"[bench] results in {args.out_dir}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
