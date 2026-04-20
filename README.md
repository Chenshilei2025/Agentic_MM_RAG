# MERP Experiment Bundle

All experiment-layer code for EMNLP submission (Route A: Unified
Framework). These files are a **subset** of the full MERP project,
isolated for repackaging into your local codebase.

## What's included (30 files)

```
experiment_bundle/
├── README.md                            ← this file
├── requirements-data.txt                ← heavy deps (pdfplumber, whisper, etc)
│
├── scripts/                             ← orchestration + analysis
│   ├── run_experiments.py               ← MAIN RUNNER (13 variants × N benchmarks)
│   ├── analyze_experiments.py           ← subset breakdown + significance tests
│   ├── modality_agnostic_ablation.py    ★ Route A CENTERPIECE (Table 2)
│   ├── eval_gpt_scorer.py               ← official MMLongBench-Doc GPT-4o scorer
│   ├── export_trace.py                  ← dump per-run decision traces
│   ├── live_demo_madqa.py               ← legacy single-benchmark demo
│   ├── run_ablation.py                  ← legacy ablation runner
│   └── baselines_external/
│       ├── mdocagent_wrapper.py         ← your main doc-side competitor
│       └── videorag_wrapper.py          ← your main video-side competitor
│
├── baselines/                           ← in-code baselines (no external deps)
│   ├── standard_rag.py
│   ├── late_fusion_rag.py
│   ├── early_fusion_rag.py
│   └── self_rag_style.py
│
├── data/                                ← data ingest layer
│   ├── parsers/
│   │   ├── schema.py                    ← ParsedBlock unified schema
│   │   ├── pdf_parser.py                ← pdfplumber + LayoutLMv3 + PaddleOCR
│   │   └── video_parser.py              ← scenedetect + whisper + ffmpeg
│   ├── embedders/
│   │   ├── bge_text.py                  ← BGE + MockBGE
│   │   └── clip_image.py                ← CLIP + MockCLIP
│   ├── index_builder.py                 ← ParsedBlocks → MultiIndexStore
│   └── loaders/
│       ├── mmlongbench_doc.py           ← primary doc benchmark
│       ├── longer_videos.py             ← primary video benchmark
│       ├── cross_corpus.py              ← Route B extension (optional)
│       └── madqa.py                     ← legacy loader (keep for reference)
│
└── experiments/
    └── EXPERIMENT_PLAN.md               ← READ THIS FIRST — full 6-week plan
```

## Integration into your local codebase

This bundle **depends on the MERP core** (orchestrator, agents, memory,
prompts, tools, cli, utils, providers, config). If you already have the
full `agentic_rag_final.tar.gz` unpacked, just drop these files in and
overwrite existing ones.

If you're merging into a larger codebase, the dependency graph is:

```
scripts/*                 depends on: baselines/, data/, memory/, orchestrator/, agents/, config/
baselines/*               depends on: memory/ (for MultiIndexStore)
data/loaders/*            depends on: data/parsers/schema.py
data/parsers/*            depends on: nothing internal (external: pdfplumber, whisper, ...)
data/embedders/*          depends on: nothing internal (external: FlagEmbedding, transformers)
data/index_builder.py     depends on: data/parsers/, data/embedders/, memory/store
```

## Quick start after integration

```bash
# 1. Install heavy deps (GPU box)
pip install -r requirements-data.txt

# 2. Smoke test (no GPU/deps required; uses mocks)
python scripts/run_experiments.py \
    --mmlongbench-root /tmp/fake \
    --use-mocks --limit 3 \
    --variants merp,late_fusion \
    --out-dir /tmp/smoke/

# 3. Read the plan
less experiments/EXPERIMENT_PLAN.md

# 4. Wire your real Anthropic + Qwen providers into:
#    scripts/live_demo_madqa.py::_build_providers

# 5. Week 3 — MMLongBench-Doc full run
python scripts/run_experiments.py \
    --mmlongbench-root /data/MMLongBench-Doc \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,late_fusion,early_fusion,self_rag_style,mdocagent \
    --limit 300 \
    --out-dir experiments/mmlbd_main/ \
    --bootstrap 1000

# 6. Week 4 — LongerVideos full run
python scripts/run_experiments.py \
    --longervideos-root /data/LongerVideos \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,late_fusion,self_rag_style,videorag \
    --limit 150 \
    --out-dir experiments/longervideos_main/ \
    --bootstrap 1000

# 7. Week 5 — produce Table 2 (Route A centerpiece)
python scripts/modality_agnostic_ablation.py \
    --doc-raw experiments/mmlbd_main/raw_results.csv \
    --video-raw experiments/longervideos_main/raw_results.csv \
    --out experiments/table2_modality_agnostic.csv \
    --include-secondary

# 8. Subset breakdown (Table 4)
python scripts/analyze_experiments.py \
    --raw experiments/mmlbd_main/raw_results.csv \
    --mmlongbench-root /data/MMLongBench-Doc \
    --out experiments/mmlbd_main/analysis/
```

## Route A paper claims (locked)

| Claim | Mechanism | Evidence |
|---|---|---|
| C1: Modality-Agnostic Decision Control | Same code on doc + video | Table 2 (modality_agnostic_ablation.py output) |
| C2: Decision-Theoretic Efficiency | VoI + InfoGain keeps tokens low | Table 3 (efficiency columns from analyze_experiments.py) |
| C3: Explicit Cross-Modal Conflict Resolution | `RESOLVE_CONFLICT` command | Table 4/5 (subset breakdown for cross-modal + disagreement cases) |

## File-by-file entry points

### `run_experiments.py`
Main runner. 13 variants (merp + 6 merp ablations + 4 baselines + 2 external
wrappers). Accepts `--mmlongbench-root`, `--longervideos-root`,
`--cross-corpus` (one or more). Produces `raw_results.csv` +
`aggregate.csv`.

### `analyze_experiments.py`
Post-hoc. Reads `raw_results.csv`. Produces:
- `subset_breakdown.csv` (Table 4/5)
- `pairwise_significance.csv` (McNemar + bootstrap for MERP vs each baseline)
- `efficiency.csv` (Table 3)

### `modality_agnostic_ablation.py` ★
Post-hoc. Reads TWO `raw_results.csv` (doc + video). For each of 3 core
mechanisms (VoI / InfoGain / Conflict), computes whether disabling it
causes significant + non-trivial drops on **both** benchmarks. Prints:
```
[table2] voi_gating:       doc Δ=0.08 p=0.001 | vid Δ=0.06 p=0.012 | agnostic=True
[table2] info_gain_sat:    doc Δ=0.03 p=0.04  | vid Δ=0.05 p=0.008 | agnostic=True
[table2] conflict_resolve: doc Δ=0.02 p=0.18  | vid Δ=0.04 p=0.02  | agnostic=False
[table2] verdict: 2/3 mechanisms validated as modality-agnostic
[table2] ✓ Route A story holds — paper contribution defensible
```
This single output is Route A's most important evidence.

### `eval_gpt_scorer.py`
Official MMLongBench-Doc 3-stage scorer (GPT-4o). Only use for final
camera-ready numbers — costs ~$0.02/query. For dev use quick_score
(built into `mmlongbench_doc.py`).

### `baselines_external/mdocagent_wrapper.py` / `videorag_wrapper.py`
Thin stubs. After `git clone` the respective repos:
```bash
export MDOCAGENT_PATH=/path/to/MDocAgent
export VIDEORAG_PATH=/path/to/VideoRAG
```
Then fill in the `run()` function. ~30 lines of adapter code each.

## Files NOT in this bundle

If you want to run, you also need these from the full MERP tar.gz:

- `main.py` (system entry)
- `agents/` (DecisionAgent, SubAgent, factory)
- `orchestrator/` (controller, voi_gating, runtime)
- `memory/` (evidence_pool, state_manager, store)
- `prompts/` (prompt_builder + skills)
- `tools/` (builtin, registry)
- `cli/` (schemas, parser, validator)
- `utils/` (info_gain_tracker, subtask_embedder, logger, asset_resolver)
- `providers/` (anthropic, deepseek, mock, base)
- `config/` (ablation, system.yaml)
- `tests/` (79 tests)

These are all in `agentic_rag_final.tar.gz`. The bundle here is strictly
the experiment delta on top of that core.

## If you want one tar.gz with EVERYTHING

```bash
tar -xzf agentic_rag_final.tar.gz        # gets the core system
tar -xzf experiment_bundle.tar.gz -C agentic_rag/   # overlays experiment layer
```

Or just use `agentic_rag_final.tar.gz` directly — it already includes
everything in this bundle.

## Test status

Tests in the core MERP tar.gz cover the baselines/ and data/loaders/
but not the experiment scripts (those are integration-level; verify by
running smoke tests above).

Run from the core repo:
```bash
cd agentic_rag
python -m tests.run_tests   # expect 79 passed, 0 failed
```

## When you come back with results

Bring:
- `experiments/mmlbd_main/aggregate.csv`
- `experiments/longervideos_main/aggregate.csv`
- `experiments/table2_modality_agnostic.csv`
- 3-5 MERP trace JSONLs
- Notes on any bizarre DA behavior

For help:
- Method section drafting
- Table caption + narrative writing
- Prompt debugging from traces
- Related work contrasting MDocAgent / HM-RAG / VideoRAG / LVAgent
- Decision on Route B extension
