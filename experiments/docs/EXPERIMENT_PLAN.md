# MERP — EMNLP Experiment Plan (Route A: Unified Framework)

**Paper positioning (LOCKED for Route A):**

> **"MERP: A Unified Decision-Theoretic Framework for Agentic Multimodal
> RAG"**
>
> We present MERP, an agentic RAG system that applies three **modality-
> agnostic** control mechanisms — Value-of-Information gating,
> information-gain saturation detection, and explicit cross-modal
> conflict resolution — to coordinate retrieval over heterogeneous
> multimodal sources. Unlike prior work that specializes either for
> documents (MDocAgent, HM-RAG, DocAgent) or for videos (VideoRAG,
> LVAgent, VideoAgent), MERP is the first unified framework demonstrated
> on both document QA and long video QA benchmarks using **the same
> codebase and the same decision mechanisms**. Extensive experiments on
> MMLongBench-Doc (1062 QA over 135 PDFs) and LongerVideos (602 queries
> over 164 videos, 134+ hours) show that MERP achieves competitive
> accuracy while using 30-50% fewer tokens than specialized baselines,
> and that its three control mechanisms all transfer significantly
> across modalities.

## Core insight that makes this pitch work

**No prior agentic RAG paper has shown one codebase works for both doc
and video.** Each paper specializes:

| Paper | Doc | Video | Unified framework? |
|---|---|---|---|
| MDocAgent (Han et al., 2025) | ✅ | ❌ | ❌ |
| HM-RAG (Liu et al., 2025, ACM MM) | ✅ text+graph | ❌ | ❌ |
| DocAgent (EMNLP 2025 main) | ✅ | ❌ | ❌ |
| VideoRAG (Ren et al., KDD 2026) | ❌ | ✅ | ❌ |
| LVAgent (ICCV 2025) | ❌ | ✅ | ❌ |
| VideoAgent (Wang et al., 2024) | ❌ | ✅ | ❌ |
| Self-RAG (ICLR 2024) | text-only | ❌ | ❌ |
| **MERP (ours)** | ✅ | ✅ | **✅** |

The paper's core argument: **a Decision-Theoretic control layer is the
right abstraction**, because the same mechanisms work for both
document QA and video QA.

---

## Three Claims (Route A locked)

| Claim | Mechanism | Main evidence |
|---|---|---|
| **C1: Modality-Agnostic Decision Control** | Same code on doc + video | **Table 2** (the key table): ablating each mechanism causes significant, non-trivial drops on BOTH benchmarks |
| **C2: Decision-Theoretic Efficiency** | VoI + InfoGain keeps costs low | Table 3: MERP achieves comparable accuracy to MDocAgent/VideoRAG at 30-50% the tokens/q |
| **C3: Explicit Cross-Modal Conflict Resolution** | `RESOLVE_CONFLICT` command | Table 4 breakdown: MERP wins on the "cross-modal" subset of MMLongBench-Doc and on "documentary" subset of LongerVideos (where ASR often contradicts visuals) |

---

## Required Tables in the paper

### Table 1a — MMLongBench-Doc (doc-only)

Direct head-to-head with MDocAgent.

| Method | Overall F1 | Text-only | Visual-only | Cross-modal | Cross-page | Unans. | tokens/q |
|---|---|---|---|---|---|---|---|
| GPT-4o (end-to-end) | 42.7 | | | | | | N/A |
| M3DocRAG | (cite) | | | | | | |
| ColBERT + LLaMA 3.1-8B | (cite) | | | | | | |
| MDocAgent | X | | | | | | Y |
| **MERP (ours)** | X | | | | | | **<Y/2** |

**Win condition(s)**:
- Ideal: overall F1 ≥ MDocAgent
- Acceptable: overall F1 within 2 pts of MDocAgent, **tokens/q is 30-50% lower**
- Backup: beat MDocAgent on cross-modal + cross-page subsets even if overall is a tie

### Table 1b — LongerVideos (video-only)

Direct head-to-head with VideoRAG.

| Method | Lectures | Documentaries | Entertainment | Avg winrate vs ref | tokens/q |
|---|---|---|---|---|---|
| VideoRAG | X | X | X | — | Y |
| LightRAG (on videos) | X | X | X | X | Y |
| GraphRAG | X | X | X | X | Y |
| **MERP (ours)** | X | X | X | X | **<Y/2** |

**Win condition(s)**:
- Ideal: winrate ≥ VideoRAG on at least 1 of 3 categories
- Acceptable: within 5% of VideoRAG on all 3, **tokens/q is 30-50% lower**

### Table 2 — Modality-Agnostic Ablation ★ (ROUTE A CENTERPIECE)

Produced by `scripts/modality_agnostic_ablation.py`. Same ablations on
both benchmarks, proving mechanisms transfer.

| Mechanism | Doc F1 (full → ablated) | Doc Δ | Doc p | Vid F1 (full → ablated) | Vid Δ | Vid p | Agnostic? |
|---|---|---|---|---|---|---|---|
| VoI gating       | A → A' | ΔA | pA | B → B' | ΔB | pB | ✓/✗ |
| InfoGain sat.    | A → A' | ΔA | pA | B → B' | ΔB | pB | ✓/✗ |
| Conflict resolve | A → A' | ΔA | pA | B → B' | ΔB | pB | ✓/✗ |

**Win condition**: ≥2 of 3 mechanisms show significant drops (p<0.05) AND
Cohen's d ≥ 0.2 on BOTH benchmarks. Script automatically computes and
prints the verdict.

### Table 3 — Efficiency (tokens / latency / steps)

Both benchmarks on one table.

| Method | MMLongBench-Doc tokens/q | MMLongBench-Doc sec/q | LongerVideos tokens/q | LongerVideos sec/q |
|---|---|---|---|---|
| MDocAgent | ~X | Y | N/A | N/A |
| VideoRAG | N/A | N/A | ~X | Y |
| Late fusion | X | Y | X | Y |
| MERP (full) | X | Y | X | Y |
| MERP w/o VoI | X | Y | X | Y |
| MERP w/o InfoGain | X | Y | X | Y |

**Win condition**: MERP tokens/q significantly less than MDocAgent on
doc AND less than VideoRAG on video. This is Route A's efficiency
story.

### Table 4 — MMLongBench-Doc subset breakdown

From `scripts/analyze_experiments.py --out subset_breakdown.csv`.

For each subset (text_only / visual_only / cross_modal / cross_page /
unanswerable), report F1 for: MERP, MDocAgent, late_fusion, each
ablation.

**Win condition**: MERP > MDocAgent on `cross_modal` and `unanswerable`.

### Table 5 — LongerVideos per-category breakdown

For each category (lectures / documentaries / entertainment), report
F1 / winrate for: MERP, VideoRAG, late_fusion, each ablation.

### Figure — Case studies (at least 2)

1. **Cross-modal conflict on doc**: one MMLongBench-Doc query where MERP
   uses `RESOLVE_CONFLICT` to pick text over a contradictory figure
   caption.
2. **Saturation on video**: one LongerVideos query where the DA
   correctly stops after N turns despite not being fully certain,
   because info-gain saturated. Show the exact turn where saturation
   fired.

---

## Datasets

### Primary (Route A)

**1. MMLongBench-Doc** (Ma et al., 2024)
- 135 PDFs, 1062 QA
- 49.4 pages avg, 20970 tokens avg per PDF
- 33% cross-page questions, 22.5% unanswerable
- Download: https://github.com/mayubo2333/MMLongBench-Doc
- Eval: 3-stage GPT-4o scorer (see `scripts/eval_gpt_scorer.py`)

**2. LongerVideos** (Ren et al., 2025)
- 164 videos, 602 queries
- 134+ hours total, 3 categories
- Download: https://github.com/HKUDS/VideoRAG (under `LongerVideos/`)
- Eval: pairwise winrate vs reference answer (GPT-4o judge)

### Optional (Route B follow-up)

**3. Cross-Corpus MM-QA** (your own)
- Build 150-250 queries pairing MMLongBench-Doc + LongerVideos
- Scripts already in `data/loaders/cross_corpus.py`
- Only pursue after Route A experiments are solid

### Additional validation (if time)

- **LongDocURL** — secondary doc benchmark, verifies MMLongBench-Doc result
  isn't dataset-specific
- **MLVU / LongVideoBench** — secondary video benchmark

---

## Backbone LLMs

Pick 2 backbones to show robustness:

| Role | Primary | Secondary |
|---|---|---|
| Decision Agent | Claude Sonnet 4 | Qwen2.5-72B-Instruct |
| Sub-agent (text) | Qwen2.5-7B-Instruct | LLaMA-3.1-8B-Instruct |
| Sub-agent (vision) | Qwen2.5-VL-7B | InternVL-2.5-8B |

Run Table 1a/1b with primary backbone. If primary shows a win, optionally
re-run a subset with secondary to prove robustness.

---

## Full experiment matrix

| # | Benchmark | Variants | Examples | Backbones | Total runs |
|---|---|---|---|---|---|
| 1 | MMLongBench-Doc | 7 MERP variants + 4 baselines + MDocAgent | 300 | 1 (primary) | 3600 |
| 2 | LongerVideos | 7 MERP variants + 2 baselines + VideoRAG | 150 | 1 (primary) | 1500 |
| 3 | Cross-benchmark backbone B spot-check (MERP only) | 1 | 100 per benchmark | 1 (secondary) | 200 |

**Total inference runs: ~5300**

Budget estimate (Claude Sonnet 4 at current rates, assuming ~8K tokens
per MERP run):
- 5300 × 8K = 42M tokens
- Input tokens dominate (~$3-5 per million)
- Estimate: **$100-200 for inference**
- GPT-4o scoring: **~$50-100** for MMLongBench-Doc (officially recommended protocol)

---

## Execution order (6-week plan)

### Week 1 — Infrastructure + smoke

- [ ] `pip install -r requirements-data.txt`
- [ ] Wire your real Anthropic + Qwen providers into
  `scripts/live_demo_madqa.py::_build_providers`
- [ ] Smoke test with mocks:
  `python scripts/run_experiments.py --mmlongbench-root /tmp/fake --use-mocks --limit 3 --out-dir /tmp/smoke/`
- [ ] Smoke test with real providers on 5 MMLongBench-Doc examples
- [ ] Smoke test on 3 LongerVideos examples (video parser is slow; verify end-to-end)

### Week 2 — Baselines

- [ ] Clone MDocAgent: `git clone https://github.com/aiming-lab/MDocAgent`
- [ ] Clone VideoRAG: `git clone https://github.com/HKUDS/VideoRAG`
- [ ] Fill in `scripts/baselines_external/mdocagent_wrapper.py::run` (~30 lines after reading their README)
- [ ] Fill in `scripts/baselines_external/videorag_wrapper.py::run`
- [ ] Run each baseline on 20 queries to verify they reproduce their paper's ballpark numbers

### Week 3 — Full MMLongBench-Doc run

```bash
python scripts/run_experiments.py \
    --mmlongbench-root /data/MMLongBench-Doc \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,merp_no_curated,merp_no_reflect,merp_no_replan,late_fusion,early_fusion,self_rag_style,standard_rag,mdocagent \
    --limit 300 \
    --out-dir experiments/mmlbd_main/ \
    --bootstrap 1000
```

### Week 4 — Full LongerVideos run

```bash
python scripts/run_experiments.py \
    --longervideos-root /data/LongerVideos \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,merp_no_curated,late_fusion,self_rag_style,videorag \
    --limit 150 \
    --out-dir experiments/longervideos_main/ \
    --bootstrap 1000
```

### Week 5 — Analysis + Table 2 (the big one)

```bash
# Subset breakdowns
python scripts/analyze_experiments.py \
    --raw experiments/mmlbd_main/raw_results.csv \
    --mmlongbench-root /data/MMLongBench-Doc \
    --out experiments/mmlbd_main/analysis/

# Modality-agnostic ablation (Route A centerpiece)
python scripts/modality_agnostic_ablation.py \
    --doc-raw experiments/mmlbd_main/raw_results.csv \
    --video-raw experiments/longervideos_main/raw_results.csv \
    --out experiments/table2_modality_agnostic.csv
```

Read the Table 2 verdict. If `modality_agnostic=True` for 2/3 core
mechanisms, your story is solid.

### Week 6 — Writing + case studies

- [ ] Export 5-10 MERP trace JSONLs from both benchmarks using
  `scripts/export_trace.py`
- [ ] Pick 2 qualitative cases for Figure
- [ ] Draft Method section (3 mechanisms + 2 tier disclosure)
- [ ] Draft Experiments section around Tables 1-5
- [ ] Draft Related Work distinguishing from MDocAgent, HM-RAG, VideoRAG, LVAgent
- [ ] Write Discussion: why modality-agnostic control matters
- [ ] Limitations section: single-backbone for primary experiments, etc.

---

## Success thresholds

### Ideal outcome (EMNLP long paper)
- MERP matches or beats MDocAgent overall F1 on MMLongBench-Doc
- MERP within 5% of VideoRAG winrate on LongerVideos
- tokens/q is 30-50% less than specialized baselines on BOTH
- 3 of 3 core mechanisms are modality-agnostic (Table 2)

### Acceptable outcome (EMNLP short paper)
- MERP within 2-3 pts of MDocAgent on doc
- Within 5% of VideoRAG on video
- tokens/q substantially less on both
- 2 of 3 core mechanisms modality-agnostic

### Fallback outcome (EMNLP Findings / COLING)
- Acceptable on one benchmark, weaker on the other
- Only 1 of 3 mechanisms transfers
- Can still report but frame as "decision control in agentic RAG;
  modality-specific adaptations discussed"

### Pivot threshold (stop and rethink)
- MERP loses by >5% on both benchmarks at similar token cost
- 0 of 3 mechanisms transfer (each doc mechanism fails on video)
- At this point the "unified framework" story is broken; consider
  submitting only to one benchmark and pitching as "efficient
  doc-agentic-RAG" or "efficient video-agentic-RAG"

---

## Command recipes

### Smoke test (no deps)
```bash
python scripts/run_experiments.py \
    --mmlongbench-root /tmp/fake \
    --use-mocks --limit 3 \
    --variants merp,late_fusion \
    --out-dir /tmp/smoke/
```

### Doc benchmark
```bash
export ANTHROPIC_API_KEY=...
python scripts/run_experiments.py \
    --mmlongbench-root /data/MMLongBench-Doc \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,late_fusion,early_fusion,self_rag_style,mdocagent \
    --limit 300 \
    --out-dir experiments/mmlbd_main/ \
    --bootstrap 1000
```

### Video benchmark
```bash
python scripts/run_experiments.py \
    --longervideos-root /data/LongerVideos \
    --variants merp,merp_no_voi,merp_no_infogain,merp_no_conflict,late_fusion,self_rag_style,videorag \
    --limit 150 \
    --out-dir experiments/longervideos_main/ \
    --bootstrap 1000
```

### Table 2 — modality-agnostic ablation (the centerpiece)
```bash
python scripts/modality_agnostic_ablation.py \
    --doc-raw experiments/mmlbd_main/raw_results.csv \
    --video-raw experiments/longervideos_main/raw_results.csv \
    --out experiments/table2.csv \
    --include-secondary
```

### Subset analysis
```bash
python scripts/analyze_experiments.py \
    --raw experiments/mmlbd_main/raw_results.csv \
    --mmlongbench-root /data/MMLongBench-Doc \
    --out experiments/mmlbd_main/analysis/
```

### Official MMLongBench-Doc scoring (GPT-4o)
Pass `--gpt-scorer` to `run_experiments.py` (triggers `eval_gpt_scorer.py`
for the official 3-stage protocol). Only do this on the final submission
run to save cost.

---

## Known risks & mitigations

| Risk | Mitigation |
|---|---|
| **MDocAgent too hard to wire** | First report their paper's numbers on MMLongBench-Doc as an honest baseline; clearly note "as reported by Han et al., 2025". Reviewers accept this if you're upfront. |
| **VideoRAG deps break** | Same: report their numbers with explicit caveats. |
| **Video parsing way too slow** | Pre-compute once, cache `.parsed.json` per video. Budget 1 day for full LongerVideos (164 videos). |
| **Table 2 doesn't validate** | 0-of-3 agnostic is a real problem. Drop to "C2 Efficiency" as primary claim, relegate mechanisms to "specialized per modality". |
| **Token cost blows up** | Start with 100-example smoke on each benchmark; scale to 300/150 only after verifying. |
| **Unanswerable subset gaming** | MMLongBench has 22% unanswerable. If MERP just says "don't know" too often, F1 on unans goes up but overall F1 tanks. Monitor this separately. |
| **Reviewer asks: "why not RL-train like LVAgent?"** | Explicit training-free positioning in the abstract: "training-free, works with any off-the-shelf backbone". This is a feature, not a weakness. |

---

## Things NOT to promise

- "Outperforms all SOTA" unless you actually do on both benchmarks
- "First agentic multimodal RAG" — MDocAgent / HM-RAG already exist
- "Cross-corpus" claims unless you build Route B
- Zero overlap with MDocAgent's 5-agent design

## Things TO emphasize

- **"First to demonstrate one unified framework on both doc AND video"** — verifiably true
- **"Three modality-agnostic mechanisms, validated by controlled ablation"** — strong empirical claim
- **"30-50% token savings"** — if it holds
- **"Training-free"** — LVAgent requires RL; you don't
- **"Interpretable decision log"** — `conflict_resolutions`, `voi_decisions`, `reflect_verdicts` are all structured + trace-able

---

## What to include in the paper repo on submission

```
merp_emnlp/
├── README.md                   (reproduction commands)
├── requirements-data.txt
├── scripts/                    (all experiment runners)
├── experiments/
│   ├── mmlbd_main/             (raw + aggregate results)
│   ├── longervideos_main/
│   ├── table2_modality_agnostic.csv / .md
│   └── analysis/
├── ...full MERP code...
└── traces/                     (selected JSONL traces for case studies)
```

---

## When you come back with results

Bring:
- `experiments/mmlbd_main/aggregate.csv`
- `experiments/longervideos_main/aggregate.csv`
- `experiments/table2_modality_agnostic.csv`
- 3-5 MERP trace JSONLs (both benchmarks, diverse queries)
- Notes on any bizarre behavior (DA looping, saturation firing too early, etc.)

I can help:
- Write the Method section
- Draft Table captions + analysis paragraphs
- Debug prompts based on trace behavior
- Write related work contrasting MDocAgent / HM-RAG / VideoRAG / LVAgent
- Decide if Route B extension is worth doing after Route A lands
