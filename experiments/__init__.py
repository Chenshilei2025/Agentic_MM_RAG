"""Experiments layer — EMNLP submission experiments.

Directory structure (organized by benchmark):
  mmlongbench/           MMLongBench-Doc benchmark experiments
  longer_videos/         LongerVideos benchmark experiments
  cross_corpus/          Cross-corpus unified retrieval experiments
  common/                Shared utilities and scripts
    ├─ runners/          Experiment runners (run_experiments, run_ablation)
    ├─ analysis/         Analysis scripts (analyze_experiments, modality_agnostic_ablation)
    ├─ utils/            Utilities (eval_gpt_scorer, export_trace, live_demo)
    └─ baselines_external/  External baseline wrappers (MDocAgent, VideoRAG)
  docs/                  Experiment plans and documentation

Usage:
    # Run experiments
    python -m experiments.common.runners.run_experiments --mmlongbench-root /data/mmlbd --variants merp,late_fusion

    # Analyze results
    python -m experiments.common.analysis.analyze_experiments --raw experiments/mmlbd/raw_results.csv

    # Modality-agnostic ablation (Route A centerpiece)
    python -m experiments.common.analysis.modality_agnostic_ablation --doc-raw experiments/mmlbd/raw_results.csv --video-raw experiments/longervideos/raw_results.csv

See docs/EXPERIMENT_PLAN.md for the full 6-week experiment plan.
"""
