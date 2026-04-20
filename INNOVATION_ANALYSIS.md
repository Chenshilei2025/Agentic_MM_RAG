# 论文创新点分析 (EMNLP 2025 Submission)

## 核心创新

### 1. 架构创新: MERP (Multi-modal Evidence Refinement Pipeline)

```
传统方法:          Query → [混合检索] → [Rerank] → Answer
                  ↓
              (模态信息丢失)

MERP:             Query → [Decompose] → aspects
                  ↓
                  aspects → Modality-Isolated Retrieval
                  ↓
                  [Tier-1: summary] → Decision Agent (Claude Opus 4.7)
                  ↓
                  [Reflect: conflict detection]
                  ↓
                  [Tier-2: curated, cross-modal filtered] ← 新增!
                  ↓
                  [Replan: gap completion]
                  ↓
                  [Tier-3: full, on-demand] ← VoI gating
                  ↓
                  Cross-modal synthesis → Answer
```

**关键差异**:
- 不是简单的early/late fusion
- 不是pipeline式固定流程
- **决策层驱动的动态多轮检索**

### 2. 渐进式披露 (Progressive Disclosure)

| Tier | 触发条件 | 内容 | Token成本 |
|------|----------|------|-----------|
| Tier-1 | 每个sub-agent | summary (structured finding) | 低 (~100 tokens) |
| Tier-2 | 跨模态conflict或重要gap | curated (top 10, conflict-boosted) | 中 (~500 tokens) |
| Tier-3 | VoI gate通过 | full raw evidence | 高 (~2k+ tokens) |

**创新**: 不是简单的"给全部证据"，而是**按需披露**:
- 无conflict → 不给Tier-2
- 非关键aspect → 不给Tier-3
- 80%+的查询只需要Tier-1

### 3. 细粒度Aspect Taxonomy

```
传统: "找关于X的证据"
MERP:  按aspect分解
       - event: "X发生了什么?"
       - spatial: "X在哪里?"
       - entity: "X涉及什么/谁?"
       - causal: "为什么X?"
       - temporal: "X什么时候?"
       - process: "X如何发生?"
```

**创新**: 
- **aspect是跨模态对齐的key** (不同模态的agent用同一aspect标签)
- enabling **cross-modal conflict detection**: 同一aspect在不同模态有矛盾结论

### 4. Sub-Agent作为Reranker

**传统**: Retrieval → Cross-Encoder (如BGE-reranker) → Top-K
**MERP**: Retrieval (raw 20) → LLM reads all → Mental rerank → Citation

**优势**:
- 视觉模态: CLIP score + LLM re-judging (看caption/图)
- Dedup是emergent: citation 5 of 20 = drop 15
- 更灵活: 能理解语义相似但不完全相同的内容

### 5. Loop Progress Tracking (Cost Control)

```
每轮记录:
- new_aspects_covered?
- conflicts_resolved?
- evidence_quality_improved?
- commands_redundant?

停滞检测 → 强制STOP/ESCALATE/REPLAN
```

**创新**: 不是简单max_steps，而是**基于进展的智能终止**:
- 3轮无进展 → consider stopping
- Evidence quality > 0.7 → STOP
- Redundant commands → force stop

## 与SOTA的区别

| Method | Retrieval | Decision | Disclosure |
|--------|-----------|----------|------------|
| Late Fusion | Separate | Separate | All evidence |
| Early Fusion | Combined | Single | All evidence |
| Self-RAG | Iterative | Single LLM | All evidence |
| **MERP** | **Modality-Isolated** | **Cross-Modal DA** | **Progressive** |

**核心差异**:
1. **Late Fusion**: 各模态独立答案 → merge. 但merge是heuristic (e.g., RRF)
2. **Early Fusion**: vector合并 → lost modality info
3. **Self-RAG**: 单LLM迭代检索 → 缺乏专门的cross-modal reasoning
4. **MERP**: 专门的Decision Agent进行cross-modal reasoning，且有progressive disclosure

## 实验设计建议

### 必须回答的问题

1. **Progressive Disclosure真的省token吗?**
   - Ablation: `merp_no_progressive_disc` vs `merp`
   - Metric: avg_tokens per query, 保持F1

2. **Cross-Modal Decision Alignment有效吗?**
   - Ablation: `merp_no_reflect` (heuristic conflict detection)
   - Case studies: 展示conflict detection例子

3. **Modality-Isolated vs Fusion?**
   - Baseline: `late_fusion`, `early_fusion`
   - Metric: F1 on cross-modal queries

4. **细粒度Aspect Taxonomy有用吗?**
   - Ablation: 粗粒度 (如 "text", "visual") vs 细粒度 (6 aspects)
   - Metric: coverage, n_aspects_covered

5. **Loop Progress Tracking防死循环?**
   - Metric: max_steps_exhausted rate, avg_iterations

### MADQA数据集适配

MADQA特点:
- Multi-document QA
- 包含图表、表格
- 需要跨文档推理

**需要确认**:
- MADQA是否有video模ality? (如没有，visual主要看图表)
- Query中跨模态比例多少?

### 建议的Baselines

1. **Standard RAG**: 单模态 (text only)
2. **Late Fusion**: 各模态检索 → RRF merge
3. **Early Fusion**: Vector拼接
4. **Self-RAG style**: 迭代检索，单LLM决策
5. **MERP**: 完整系统
6. **MERP ablations**:
   - No Reflect
   - No Replan
   - No Progressive Disclosure
   - No Info Gain (saturation check)

### 预期结果

| Method | EM | F1 | Tokens |
|--------|----|----|----|
| Standard RAG | - | - | - |
| Late Fusion | - | - | - |
| Early Fusion | - | - | - |
| Self-RAG | - | - | - |
| MERP (full) | **↑** | **↑** | **↓** |
| - No Reflect | ↓ | ↓ | similar |
| - No Progressive | similar | similar | **↑↑** |

**核心claim**: MERP在保持/提升质量的同时，显著降低token成本

## 论文结构建议

### Title
"MERP: Multi-modal Evidence Refinement with Progressive Disclosure and Cross-Modal Decision Alignment"

### Abstract
- Multi-modal RAG challenge
- Modality-isolated retrieval + cross-modal decision
- Progressive three-tier disclosure
- Sub-agent as reranker
- Results: ↑F1, ↓tokens on MADQA

### Introduction
- Multi-modal RAG的重要性
- 现有方法的局限 (fusion方式, 成本)
- 我们的贡献 (4点)

### Method
1. Overview (Figure 1)
2. Modality-Isolated Retrieval
3. Progressive Three-Tier Disclosure
4. Decision Agent Skills (Decompose/Reflect/Replan)
5. Sub-Agent as Reranker
6. Loop Progress Tracking

### Experiments
- Setup (MADQA, metrics, baselines)
- Main Results (Table 1)
- Ablation Studies (Table 2)
- Case Studies (conflict detection examples)
- Token Analysis (Table 3)

### Discussion
- When does progressive disclosure help?
- Cross-modal alignment impact
- Cost-quality tradeoff

## 需要补充的实验

当前 `scripts/run_ablation.py` 已有:
- ✅ Baselines (standard, late_fusion, early_fusion, self_rag_style)
- ✅ MERP ablations (no_reflect, no_replan, no_info_gain)
- ✅ Metrics (EM, F1, tokens)

**建议添加**:
- Ablation: `no_progressive_disc` (绕过Tier-2 gate, 总是给full evidence)
- Ablation: coarse_grained_aspects (如只用 "text", "visual")
- Token breakdown per-tier (证明progressive disclosure省在哪儿)

### Case Study收集

在 `live_demo.py` 中添加 `--case-study` flag:
- 记录每个aspect的跨模态证据
- 记录conflict detection过程
- 记录replan触发情况
- 输出detailed trace用于论文figure
