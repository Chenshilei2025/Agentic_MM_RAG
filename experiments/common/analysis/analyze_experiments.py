#!/usr/bin/env python3
"""Post-hoc analysis of raw_results.csv produced by run_experiments.py.

Produces:
  - Table 5 data: performance broken down by question-type subsets
    (text_only / visual_only / cross_modal / cross_page / unanswerable /
    both_needed / complementary / conflicting)
  - Pairwise significance tests: paired McNemar on EM, bootstrap test on F1
  - Effect size: Cohen's d for F1 differences
  - Win-rate tables for cross-corpus (MERP vs each baseline)

Usage:
  python scripts/analyze_experiments.py \
      --raw experiments/run_2026_04/raw_results.csv \
      --mmlongbench-root /data/MMLongBench-Doc \
      --cross-corpus /data/cross_corpus.json \
      --out experiments/run_2026_04/analysis/
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Significance tests
# ===========================================================================
def mcnemar_test(a_correct: List[int], b_correct: List[int]
                  ) -> Tuple[float, float, int, int]:
    """Paired McNemar's test for two models' EM outputs on the same queries.

    Returns (statistic, p_value, n_a_only_correct, n_b_only_correct).
    p-value uses the continuity-corrected chi-square approximation. For
    small counts (< 25) we fall back to exact binomial.
    """
    assert len(a_correct) == len(b_correct), "length mismatch"
    b01 = sum(1 for a, b in zip(a_correct, b_correct) if a == 0 and b == 1)
    b10 = sum(1 for a, b in zip(a_correct, b_correct) if a == 1 and b == 0)

    if b01 + b10 < 25:
        # Exact binomial test: null H = 0.5 split
        n, k = b01 + b10, min(b01, b10)
        # Two-sided exact p-value
        from math import comb
        def pmf(k, n, p=0.5):
            return comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        p_val = 2 * sum(pmf(i, n) for i in range(k + 1))
        p_val = min(1.0, p_val)
        return float(k), p_val, b01, b10

    # Large-sample chi-square with continuity correction
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    # chi-square df=1 → p = erfc(sqrt(stat/2))
    from math import erfc, sqrt
    p_val = erfc(sqrt(stat / 2))
    return stat, p_val, b01, b10


def bootstrap_paired_diff(a_vals: List[float], b_vals: List[float],
                           n_boot: int = 10000, seed: int = 0
                           ) -> Tuple[float, float, float, float]:
    """Paired bootstrap test on mean(A) - mean(B).

    Returns (mean_diff, ci_lo, ci_hi, p_value_two_sided).
    """
    import random
    rng = random.Random(seed)
    n = len(a_vals)
    assert n == len(b_vals)
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    observed = sum(diffs) / n

    boot_means = []
    for _ in range(n_boot):
        boot = [diffs[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(boot) / n)
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]

    # two-sided p-value: fraction of bootstrap means crossing 0
    if observed >= 0:
        p_val = 2 * sum(1 for x in boot_means if x <= 0) / n_boot
    else:
        p_val = 2 * sum(1 for x in boot_means if x >= 0) / n_boot
    p_val = min(1.0, p_val)

    return observed, lo, hi, p_val


def cohens_d(a_vals: List[float], b_vals: List[float]) -> float:
    """Paired Cohen's d (standardized mean difference)."""
    diffs = [a - b for a, b in zip(a_vals, b_vals)]
    n = len(diffs)
    if n < 2:
        return 0.0
    m = sum(diffs) / n
    var = sum((x - m) ** 2 for x in diffs) / (n - 1)
    sd = math.sqrt(var)
    return m / sd if sd > 0 else 0.0


# ===========================================================================
# Data loaders for subset tagging
# ===========================================================================
def load_mmlongbench_tags(root: str) -> Dict[str, Dict[str, Any]]:
    """Return {query_id: {subsets...}} for MMLongBench-Doc."""
    from data.loaders.mmlongbench_doc import load_mmlongbench_doc
    examples, _ = load_mmlongbench_doc(root)
    tags: Dict[str, Dict[str, Any]] = {}
    for e in examples:
        tags[e.id] = {
            "text_only":   e.has_text_evidence and not e.has_visual_evidence,
            "visual_only": e.has_visual_evidence and not e.has_text_evidence,
            "cross_modal": e.has_text_evidence and e.has_visual_evidence,
            "cross_page":  e.is_cross_page,
            "unanswerable": e.is_unanswerable,
        }
    return tags


def load_cross_corpus_tags(path: str) -> Dict[str, Dict[str, Any]]:
    from data.loaders.cross_corpus import load_cross_corpus
    examples = load_cross_corpus(path)
    tags: Dict[str, Dict[str, Any]] = {}
    for e in examples:
        tags[e.id] = {
            "both_needed":   e.relation == "both_needed",
            "complementary": e.relation == "complementary",
            "conflicting":   e.relation == "conflicting",
        }
    return tags


# ===========================================================================
# Analysis
# ===========================================================================
def analyze(raw_csv: str, tags_by_bench: Dict[str, Dict[str, Dict[str, Any]]],
             out_dir: str, reference_variant: str = "merp"):
    """Produce all analysis outputs."""
    os.makedirs(out_dir, exist_ok=True)

    # Index raw rows by (benchmark, variant, query_id)
    rows_by_qv: Dict[Tuple[str, str, str], Dict] = {}
    bench_queries: Dict[str, List[str]] = defaultdict(list)
    variants_by_bench: Dict[str, set] = defaultdict(set)

    with open(raw_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            bench, variant, qid = r["benchmark"], r["variant"], r["query_id"]
            rows_by_qv[(bench, variant, qid)] = r
            if qid not in bench_queries[bench]:
                bench_queries[bench].append(qid)
            variants_by_bench[bench].add(variant)

    # --- Table 5: subset breakdown ---
    subset_table: List[Dict[str, Any]] = []
    for bench, qids in bench_queries.items():
        tags = tags_by_bench.get(bench, {})
        if not tags:
            continue
        # discover subset names from one example tag
        subset_names = sorted({k for v in tags.values() for k in v})
        for variant in sorted(variants_by_bench[bench]):
            for subset in subset_names:
                relevant = [q for q in qids
                            if tags.get(q, {}).get(subset)]
                if not relevant:
                    continue
                ems = [int(rows_by_qv[(bench, variant, q)]["em"])
                       for q in relevant
                       if (bench, variant, q) in rows_by_qv]
                f1s = [float(rows_by_qv[(bench, variant, q)]["f1"])
                       for q in relevant
                       if (bench, variant, q) in rows_by_qv]
                if not ems:
                    continue
                subset_table.append({
                    "benchmark": bench, "variant": variant,
                    "subset": subset, "n": len(ems),
                    "em": round(sum(ems) / len(ems), 4),
                    "f1": round(sum(f1s) / len(f1s), 4)})

    if subset_table:
        subset_csv = os.path.join(out_dir, "subset_breakdown.csv")
        with open(subset_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(subset_table[0].keys()))
            w.writeheader()
            w.writerows(subset_table)
        print(f"[analyze] subset breakdown: {subset_csv}")

    # --- Pairwise significance: MERP vs each other variant ---
    sig_rows: List[Dict[str, Any]] = []
    for bench, qids in bench_queries.items():
        if reference_variant not in variants_by_bench[bench]:
            continue
        ref_ems = []
        ref_f1s = []
        aligned_qids = []
        for q in qids:
            if (bench, reference_variant, q) not in rows_by_qv:
                continue
            ref_ems.append(int(rows_by_qv[(bench, reference_variant, q)]["em"]))
            ref_f1s.append(float(rows_by_qv[(bench, reference_variant, q)]["f1"]))
            aligned_qids.append(q)

        for other in sorted(variants_by_bench[bench]):
            if other == reference_variant:
                continue
            other_ems, other_f1s = [], []
            for q in aligned_qids:
                if (bench, other, q) not in rows_by_qv:
                    break
                other_ems.append(int(rows_by_qv[(bench, other, q)]["em"]))
                other_f1s.append(float(rows_by_qv[(bench, other, q)]["f1"]))
            if len(other_ems) != len(ref_ems):
                continue
            stat, p_em, b01, b10 = mcnemar_test(ref_ems, other_ems)
            diff, lo, hi, p_f1 = bootstrap_paired_diff(ref_f1s, other_f1s)
            d = cohens_d(ref_f1s, other_f1s)
            sig_rows.append({
                "benchmark": bench,
                "reference": reference_variant, "other": other,
                "n": len(ref_ems),
                "ref_wins": b10,       # ref correct, other wrong
                "other_wins": b01,     # ref wrong, other correct
                "em_p": round(p_em, 6),
                "f1_diff": round(diff, 4),
                "f1_ci_lo": round(lo, 4), "f1_ci_hi": round(hi, 4),
                "f1_p": round(p_f1, 6),
                "cohens_d": round(d, 3),
                "significant_at_0.05": p_em < 0.05 and p_f1 < 0.05})
    if sig_rows:
        sig_csv = os.path.join(out_dir, "pairwise_significance.csv")
        with open(sig_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(sig_rows[0].keys()))
            w.writeheader()
            w.writerows(sig_rows)
        print(f"[analyze] pairwise significance: {sig_csv}")

    # --- Token efficiency table (Table 4) ---
    eff_rows = []
    variants_all = set()
    for b in variants_by_bench.values():
        variants_all |= b
    for bench in bench_queries:
        for variant in sorted(variants_all):
            ems, f1s, toks, secs = [], [], [], []
            for q in bench_queries[bench]:
                r = rows_by_qv.get((bench, variant, q))
                if r is None:
                    continue
                ems.append(int(r["em"]))
                f1s.append(float(r["f1"]))
                toks.append(int(r["n_tokens"]))
                secs.append(float(r["elapsed_sec"]))
            if not ems:
                continue
            em_mean = sum(ems) / len(ems)
            tok_mean = sum(toks) / len(toks)
            eff_rows.append({
                "benchmark": bench, "variant": variant, "n": len(ems),
                "em": round(em_mean, 4),
                "f1": round(sum(f1s) / len(f1s), 4),
                "avg_tokens": round(tok_mean, 0),
                "em_per_1ktok": round(em_mean / max(tok_mean / 1000, 1e-9), 4),
                "avg_sec": round(sum(secs) / len(secs), 2)})
    if eff_rows:
        eff_csv = os.path.join(out_dir, "efficiency.csv")
        with open(eff_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(eff_rows[0].keys()))
            w.writeheader()
            w.writerows(eff_rows)
        print(f"[analyze] efficiency: {eff_csv}")

    print(f"\n[analyze] all outputs in {out_dir}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, help="raw_results.csv")
    p.add_argument("--mmlongbench-root", default=None)
    p.add_argument("--cross-corpus", default=None)
    p.add_argument("--reference", default="merp",
                   help="reference variant for pairwise comparisons")
    p.add_argument("--out", required=True, help="output dir")
    return p.parse_args()


def main():
    args = parse_args()
    tags_by_bench = {}
    if args.mmlongbench_root:
        tags_by_bench["mmlongbench"] = load_mmlongbench_tags(args.mmlongbench_root)
    if args.cross_corpus:
        tags_by_bench["cross_corpus"] = load_cross_corpus_tags(args.cross_corpus)
    analyze(args.raw, tags_by_bench, args.out,
            reference_variant=args.reference)


if __name__ == "__main__":
    main()
