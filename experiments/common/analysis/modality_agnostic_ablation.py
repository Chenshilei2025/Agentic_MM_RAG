#!/usr/bin/env python3
"""Modality-Agnostic Ablation Analysis — the CENTERPIECE of Route A.

This script produces Table 2 of the paper: evidence that MERP's three
control mechanisms (VoI gating, information-gain saturation, explicit
cross-modal conflict resolution) are **modality-agnostic** — the SAME
ablation causes consistent drops on BOTH the doc benchmark
(MMLongBench-Doc) and the video benchmark (LongerVideos).

This is Route A's strongest selling point: "One mechanism, works for
docs AND videos." If the ablation pattern transfers across modalities,
you have a unified framework story. If it doesn't, you probably need
different mechanisms per modality (and your paper's contribution is
weaker).

Output CSV (one row per mechanism):

  mechanism, doc_f1_full, doc_f1_ablated, doc_drop, doc_p,
             vid_f1_full, vid_f1_ablated, vid_drop, vid_p,
             modality_agnostic, cohens_d_doc, cohens_d_vid

modality_agnostic=True when:
  - both benchmarks show statistically significant drop (p<0.05)
  - signs of both drops are the same (both positive)
  - Cohen's d ≥ 0.2 on both (non-trivial effect)

Usage:
  python scripts/modality_agnostic_ablation.py \\
      --doc-raw experiments/route_a/mmlbd/raw_results.csv \\
      --video-raw experiments/route_a/longervideos/raw_results.csv \\
      --out experiments/route_a/table2_modality_agnostic.csv
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Reuse paired tests from analyze_experiments
from experiments.common.analysis.analyze_experiments import (mcnemar_test, bootstrap_paired_diff,
                                                             cohens_d)


# The 3 ablations that define Route A's modality-agnostic story.
# Map mechanism name → variant that disables it.
CORE_MECHANISMS = [
    ("voi_gating",       "merp", "merp_no_voi"),
    ("info_gain_sat",    "merp", "merp_no_infogain"),
    ("conflict_resolve", "merp", "merp_no_conflict"),
]

# Additional (not claim-critical but nice to see):
SECONDARY_MECHANISMS = [
    ("tier2_curated",    "merp", "merp_no_curated"),
    ("reflect_skill",    "merp", "merp_no_reflect"),
    ("replan_skill",     "merp", "merp_no_replan"),
]


def load_raw_csv(path: str) -> Dict[Tuple[str, str], Dict[str, float]]:
    """Returns {(variant, query_id): {em, f1, n_tokens, elapsed_sec}}."""
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["variant"], r["query_id"])] = {
                "em": int(r["em"]),
                "f1": float(r["f1"]),
                "n_tokens": int(r["n_tokens"]),
                "elapsed_sec": float(r["elapsed_sec"])}
    return out


def paired_on_queries(index: Dict[Tuple[str, str], Dict[str, float]],
                       var_a: str, var_b: str,
                       metric: str = "f1"
                       ) -> Tuple[List[float], List[float]]:
    """Return aligned (a_values, b_values) for queries where BOTH
    variants produced a result."""
    qids_a = {q for (v, q) in index if v == var_a}
    qids_b = {q for (v, q) in index if v == var_b}
    shared = sorted(qids_a & qids_b)
    a_vals = [index[(var_a, q)][metric] for q in shared]
    b_vals = [index[(var_b, q)][metric] for q in shared]
    return a_vals, b_vals, shared


def compute_row(mechanism_name: str,
                 doc_idx: Dict, video_idx: Dict,
                 full_variant: str, ablated_variant: str,
                 n_boot: int = 1000) -> Dict[str, Any]:
    """Compute one mechanism's cross-benchmark ablation row."""
    # Doc side
    if doc_idx is not None:
        doc_full, doc_abl, _ = paired_on_queries(
            doc_idx, full_variant, ablated_variant, "f1")
        doc_full_em, doc_abl_em, _ = paired_on_queries(
            doc_idx, full_variant, ablated_variant, "em")
        if doc_full and doc_abl:
            doc_full_mean = sum(doc_full) / len(doc_full)
            doc_abl_mean = sum(doc_abl) / len(doc_abl)
            doc_drop = doc_full_mean - doc_abl_mean
            _, _, _, doc_p = bootstrap_paired_diff(doc_full, doc_abl, n_boot)
            doc_d = cohens_d(doc_full, doc_abl)
            _, doc_em_p, _, _ = mcnemar_test(doc_full_em, doc_abl_em)
        else:
            doc_full_mean = doc_abl_mean = doc_drop = doc_p = doc_d = doc_em_p = float("nan")
    else:
        doc_full_mean = doc_abl_mean = doc_drop = doc_p = doc_d = doc_em_p = float("nan")

    # Video side
    if video_idx is not None:
        vid_full, vid_abl, _ = paired_on_queries(
            video_idx, full_variant, ablated_variant, "f1")
        vid_full_em, vid_abl_em, _ = paired_on_queries(
            video_idx, full_variant, ablated_variant, "em")
        if vid_full and vid_abl:
            vid_full_mean = sum(vid_full) / len(vid_full)
            vid_abl_mean = sum(vid_abl) / len(vid_abl)
            vid_drop = vid_full_mean - vid_abl_mean
            _, _, _, vid_p = bootstrap_paired_diff(vid_full, vid_abl, n_boot)
            vid_d = cohens_d(vid_full, vid_abl)
            _, vid_em_p, _, _ = mcnemar_test(vid_full_em, vid_abl_em)
        else:
            vid_full_mean = vid_abl_mean = vid_drop = vid_p = vid_d = vid_em_p = float("nan")
    else:
        vid_full_mean = vid_abl_mean = vid_drop = vid_p = vid_d = vid_em_p = float("nan")

    # Modality-agnostic verdict
    def _sig(x): return isinstance(x, float) and x < 0.05
    def _nontrivial(d): return isinstance(d, float) and d >= 0.2
    modality_agnostic = (
        _sig(doc_p) and _sig(vid_p) and
        doc_drop > 0 and vid_drop > 0 and
        _nontrivial(doc_d) and _nontrivial(vid_d))

    return {
        "mechanism": mechanism_name,
        "ablated_variant": ablated_variant,
        # Doc column block
        "doc_n": len(doc_full) if doc_idx and doc_full else 0,
        "doc_f1_full": round(doc_full_mean, 4) if isinstance(doc_full_mean, float) else "",
        "doc_f1_ablated": round(doc_abl_mean, 4) if isinstance(doc_abl_mean, float) else "",
        "doc_drop": round(doc_drop, 4) if isinstance(doc_drop, float) else "",
        "doc_f1_p": round(doc_p, 6) if isinstance(doc_p, float) else "",
        "doc_em_p": round(doc_em_p, 6) if isinstance(doc_em_p, float) else "",
        "doc_cohens_d": round(doc_d, 3) if isinstance(doc_d, float) else "",
        # Video column block
        "vid_n": len(vid_full) if video_idx and vid_full else 0,
        "vid_f1_full": round(vid_full_mean, 4) if isinstance(vid_full_mean, float) else "",
        "vid_f1_ablated": round(vid_abl_mean, 4) if isinstance(vid_abl_mean, float) else "",
        "vid_drop": round(vid_drop, 4) if isinstance(vid_drop, float) else "",
        "vid_f1_p": round(vid_p, 6) if isinstance(vid_p, float) else "",
        "vid_em_p": round(vid_em_p, 6) if isinstance(vid_em_p, float) else "",
        "vid_cohens_d": round(vid_d, 3) if isinstance(vid_d, float) else "",
        # Verdict
        "modality_agnostic": modality_agnostic,
    }


def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Render a publication-ready markdown table."""
    header = ("| Mechanism | Doc F1 (full → ablated) | Doc ΔF1 | Doc p | "
              "Vid F1 (full → ablated) | Vid ΔF1 | Vid p | Agnostic |")
    sep = "|" + "|".join(["---"] * 8) + "|"
    lines = [header, sep]
    for r in rows:
        doc_cell = f"{r['doc_f1_full']} → {r['doc_f1_ablated']}" if r['doc_f1_full'] != "" else "—"
        vid_cell = f"{r['vid_f1_full']} → {r['vid_f1_ablated']}" if r['vid_f1_full'] != "" else "—"
        lines.append(
            f"| {r['mechanism']} | {doc_cell} | {r['doc_drop']} | "
            f"{r['doc_f1_p']} | {vid_cell} | {r['vid_drop']} | "
            f"{r['vid_f1_p']} | {'✓' if r['modality_agnostic'] else '✗'} |")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser(
        description="Produce the modality-agnostic ablation table. This "
                    "is Table 2 of the paper and the single most important "
                    "piece of evidence for Route A's positioning.")
    p.add_argument("--doc-raw", required=True,
                   help="raw_results.csv from doc benchmark (MMLongBench-Doc)")
    p.add_argument("--video-raw", required=True,
                   help="raw_results.csv from video benchmark (LongerVideos)")
    p.add_argument("--out", required=True, help="output CSV path")
    p.add_argument("--include-secondary", action="store_true",
                   help="also include curated/reflect/replan mechanisms")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="bootstrap samples per significance test")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"[table2] loading doc results from {args.doc_raw}")
    doc_idx = load_raw_csv(args.doc_raw)
    doc_variants = {v for (v, _) in doc_idx}
    print(f"[table2]   doc variants: {sorted(doc_variants)}")

    print(f"[table2] loading video results from {args.video_raw}")
    video_idx = load_raw_csv(args.video_raw)
    video_variants = {v for (v, _) in video_idx}
    print(f"[table2]   video variants: {sorted(video_variants)}")

    mechanisms = list(CORE_MECHANISMS)
    if args.include_secondary:
        mechanisms.extend(SECONDARY_MECHANISMS)

    rows = []
    for name, full_v, ablated_v in mechanisms:
        if full_v not in doc_variants and full_v not in video_variants:
            print(f"[table2] SKIP {name}: full variant '{full_v}' missing")
            continue
        if ablated_v not in doc_variants and ablated_v not in video_variants:
            print(f"[table2] SKIP {name}: ablated variant '{ablated_v}' missing")
            continue
        row = compute_row(
            name,
            doc_idx if full_v in doc_variants else None,
            video_idx if full_v in video_variants else None,
            full_v, ablated_v, n_boot=args.bootstrap)
        rows.append(row)
        print(f"[table2] {name}: doc Δ={row['doc_drop']} p={row['doc_f1_p']} "
              f"| vid Δ={row['vid_drop']} p={row['vid_f1_p']} "
              f"| agnostic={row['modality_agnostic']}")

    if not rows:
        print("[table2] no rows computed; check variant names", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[table2] CSV written to {args.out}")

    # Also write the markdown version
    md_path = args.out.replace(".csv", ".md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Table 2: Modality-Agnostic Ablation\n\n")
        f.write("Each row shows how disabling ONE control mechanism affects "
                "both benchmarks.\n")
        f.write("A mechanism is 'modality-agnostic' if the drop is significant "
                "(p<0.05) and Cohen's d≥0.2 on BOTH benchmarks.\n\n")
        f.write(render_markdown_table(rows))
        f.write("\n")
    print(f"[table2] markdown: {md_path}")

    # Final verdict summary
    n_agnostic = sum(1 for r in rows if r["modality_agnostic"])
    print(f"\n[table2] verdict: {n_agnostic}/{len(rows)} mechanisms "
          f"validated as modality-agnostic")
    if n_agnostic >= 2:
        print("[table2] ✓ Route A story holds — paper contribution defensible")
    elif n_agnostic == 1:
        print("[table2] ⚠ only 1/3 mechanisms transfer; consider narrowing the claim")
    else:
        print("[table2] ✗ no mechanism transfers — Route A positioning needs rework")


if __name__ == "__main__":
    main()
