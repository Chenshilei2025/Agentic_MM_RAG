# Experiments Directory

This directory contains all experiments for the EMNLP submission, organized by benchmark.

## Directory Structure

```
experiments/
├── mmlongbench/          # MMLongBench-Doc benchmark experiments
├── longer_videos/        # LongerVideos benchmark experiments
├── cross_corpus/         # Cross-corpus unified retrieval experiments (Route B)
├── common/               # Shared utilities and scripts
│   ├── runners/          # Experiment runners
│   ├── analysis/         # Analysis scripts
│   ├── utils/            # Utilities (scorers, exporters, demos)
│   └── baselines_external/  # External baseline wrappers
└── docs/                 # Experiment documentation
    ├── EXPERIMENT_PLAN.md
    └── requirements-data.txt
```

## Quick Start

### Run Experiments

```bash
# Route A: MMLongBench-Doc
python -m experiments.common.runners.run_experiments \
    --mmlongbench-root /data/MMLongBench-Doc \
    --variants merp,late_fusion,early_fusion,self_rag_style \
    --out-dir experiments/mmlongbench/results/

# Route A: LongerVideos
python -m experiments.common.runners.run_experiments \
    --longervideos-root /data/LongerVideos \
    --variants merp,late_fusion,self_rag_style,videorag \
    --out-dir experiments/longer_videos/results/
```

### Analyze Results

```bash
# Subset breakdown + significance tests
python -m experiments.common.analysis.analyze_experiments \
    --raw experiments/mmlongbench/results/raw_results.csv \
    --mmlongbench-root /data/MMLongBench-Doc \
    --out experiments/mmlongbench/analysis/

# Modality-agnostic ablation (Route A centerpiece - Table 2)
python -m experiments.common.analysis.modality_agnostic_ablation \
    --doc-raw experiments/mmlongbench/results/raw_results.csv \
    --video-raw experiments/longer_videos/results/raw_results.csv \
    --out experiments/table2_modality_agnostic.csv
```

### Evaluate with GPT-4o

```bash
# Official MMLongBench-Doc scorer (camera-ready only)
python -m experiments.common.utils.eval_gpt_scorer \
    --predictions experiments/mmlongbench/results/predictions.json \
    --ground-truth /data/MMLongBench-Doc/annotations.json \
    --out experiments/mmlongbench/results/gpt4o_scores.json
```

## Paper Tables

- **Table 1a/b**: Main benchmark results (MMLongBench-Doc, LongerVideos)
- **Table 2**: Modality-agnostic ablation (use `modality_agnostic_ablation.py`)
- **Table 3**: Efficiency metrics (tokens, latency)
- **Table 4/5**: Subset breakdown (use `analyze_experiments.py`)

## See Also

- `docs/EXPERIMENT_PLAN.md` - Full 6-week experiment plan
- `../PROJECT_STRUCTURE.md` - Overall project structure
