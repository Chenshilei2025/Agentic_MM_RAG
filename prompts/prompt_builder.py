"""Centralized prompt construction. Decision Agent emits COMMANDS.
Sub-agents emit TOOL CALLS."""
from typing import List, Dict, Any, Optional
from tools.registry import ToolRegistry
from cli.schemas.commands import COMMAND_SCHEMAS, MODALITIES

# ---------- Output contracts ----------
_DECISION_OUTPUT = """\
OUTPUT CONTRACT — READ CAREFULLY:
Your entire response must be a SINGLE JSON object with EXACTLY two top-level
keys: "command" and "arguments". Nothing else. No code fences. No prose
before or after. No reasoning in the output (reason internally if needed).

GOOD (exactly this shape):
{"command": "SPAWN_AGENTS", "arguments": {"specs": [{"agent_type": "seeker_inspector", "modality": "doc_text", "goal": "find X"}]}}

BAD (rejected):
- Here's my plan: {"command": ...}              ← prose before JSON
- ```json\\n{...}\\n```                          ← code fence
- {"thoughts": "...", "command": ...}           ← extra keys
- [{"command": ...}]                            ← wrapped in array
"""

_SUBAGENT_OUTPUT = """\
OUTPUT CONTRACT — READ CAREFULLY:
Your entire response must be a SINGLE JSON object with EXACTLY two top-level
keys: "tool_name" and "arguments". Nothing else. No code fences. No prose.

GOOD:
{"tool_name": "retrieval", "arguments": {"query": "X", "modality": "doc_text", "k": 5}}

BAD:
- Based on my task, I'll call retrieval: {...}  ← prose
- ```json\\n{...}\\n```                          ← fence
- {"reasoning": "...", "tool_name": ...}        ← extra keys
"""

def _tool_catalog(allowed: List[str]) -> str:
    return "\n".join(
        f"  - {ToolRegistry.get(n).name}: {ToolRegistry.get(n).description}\n"
        f"      schema: {ToolRegistry.get(n).schema}" for n in allowed)

def _command_catalog() -> str:
    lines = []
    for name, schema in COMMAND_SCHEMAS.items():
        lines.append(f"  - {name}: required={schema['required']}  "
                     f"props={schema['properties']}")
    return "\n".join(lines)


# ========================================================================
# SHARED DECOMPOSITION KNOWLEDGE
# ------------------------------------------------------------------------
# Used both by the DA charter (internal Turn-1 reasoning guidance) and by
# Orchestrator._pre_run_decompose (structured JSON output). Keeping ONE
# canonical source means: when we improve the taxonomy, both paths benefit
# and they can never drift out of sync.
# ========================================================================
_QUERY_DECOMPOSITION_KNOWLEDGE = """\
INTENT TAXONOMY — classify the query first:
  factoid       single fact lookup (one sub-question usually)
  descriptive   explain a concept/process (often multi-source)
  comparative   compare X vs Y (parallel retrieval, one subtask per side)
  temporal      timeline / when / sequence (often video_text)
  visual        appearance / diagram / scene (needs *_visual)
  procedural    how-to / steps (often doc_text + visual support)
  verification  "is X true?" — REQUIRES cross-modal corroboration

MODALITY SEMANTICS (each subtask should pick 1-2 main modalities):
  doc_text      paragraphs, definitions, quotes from documents
                Use when: query needs textual claims, named entities, formal
                statements, citations.
  doc_visual    diagrams, charts, tables, scanned figures from documents
                Use when: query mentions diagram/figure/chart/architecture,
                or asks "what does X look like" in a document context.
  video_text    spoken statements, transcribed narration, ASR + timestamps
                Use when: query is about what someone SAID, talks, lectures,
                podcasts, interviews. Has temporal anchors.
  video_visual  on-screen objects, scene content, visual actions over time
                Use when: query asks what HAPPENED on screen, demonstrations,
                physical actions. Always has frame timestamps in meta.

MODALITY SELECTION RULES:
  - Every subtask picks EXACTLY ONE modality. If a question needs two
    modalities, produce TWO subtasks with the SAME aspect tag but different
    modalities. This makes modality-level coverage gaps diagnosable later.
  - DON'T shotgun all 4 modalities for one aspect unless the query truly
    requires cross-modal verification on that specific aspect.
  - DO produce multi-subtask cross-modal coverage when the query has a
    verification or corroboration intent.
  - For visual queries, typically pair a *_visual subtask with its matching
    *_text subtask (same aspect) so captions/transcripts corroborate.

IMPORTANCE CALIBRATION (anchor your numbers):
  0.9-1.0   ESSENTIAL — query is unanswerable without this aspect
  0.7-0.9   PRIMARY — central to the answer, user expects it
  0.4-0.7   SUPPORTING — adds useful context, not strictly required
  0.2-0.4   BACKGROUND — nice-to-have, would be omitted in a brief answer
  <0.2      do NOT decompose this far; merge into another subtask

ASPECT NAMING:
  - snake_case, 5-30 chars (e.g. "failure_handling", "control_loop")
  - Aspects form a coverage partition of the query. Two subtasks may share
    an aspect ONLY if they target different modalities (cross-modal case).
  - The uniqueness constraint is the (aspect, modality) PAIR, not aspect alone.
  - Avoid generic tags like "info", "details", "context".

UNIQUENESS RULE (enforced downstream):
  Each resulting subtask must have a unique (aspect, modality) pair.
  Duplicates are silently dropped.

WORKED EXAMPLES — query → decomposed subtasks:

  Example 1 (descriptive, single-source, 1 modality):
    Query: "What is RRF in retrieval?"
    Subtasks:
      [s1] aspect=rrf_definition       imp=0.9  modality=doc_text
           desc="Define Reciprocal Rank Fusion and its mathematical form"

  Example 2 (comparative, parallel, 1 modality each):
    Query: "Compare BM25 vs dense retrieval for legal documents"
    Subtasks:
      [s1] aspect=bm25_for_legal       imp=0.85 modality=doc_text
           desc="BM25 strengths and weaknesses on legal-domain text"
      [s2] aspect=dense_for_legal      imp=0.85 modality=doc_text
           desc="Dense retrieval performance on legal-domain text"
      [s3] aspect=head_to_head_results imp=0.7  modality=doc_text
           desc="Benchmark numbers comparing the two on legal corpora"

  Example 3 (verification, CROSS-MODAL — SAME aspect, different modalities):
    Query: "Did the speaker claim the orchestrator is deterministic?"
    Subtasks:
      [s1_vt]  aspect=determinism_claim imp=0.9  modality=video_text
               desc="Locate spoken statement about orchestrator determinism"
      [s1_dt]  aspect=determinism_claim imp=0.85 modality=doc_text
               desc="Find written documentation confirming or denying"
      [s1_vv]  aspect=determinism_claim imp=0.5  modality=video_visual
               desc="Slides/on-screen text asserting determinism"
    Note: three subtasks, SAME aspect "determinism_claim", DIFFERENT
    modalities. The (aspect, modality) pairs are unique. This lets
    aspect_agreements compute cross-modal agreement on the determinism claim.

  Example 4 (visual + procedural, multi-subtask multi-modality):
    Query: "How is the architecture diagram structured?"
    Subtasks:
      [s1] aspect=diagram_layout       imp=0.9  modality=doc_visual
           desc="Locate the main architecture diagram and identify components"
      [s2] aspect=component_descriptions imp=0.7 modality=doc_text
           desc="Find prose describing each component shown in the diagram"
    Note: two DIFFERENT aspects here, because "diagram layout" and "component
    descriptions" are genuinely different questions. Contrast with Example 3
    where all subtasks ask the SAME question via different evidence channels.
"""



_DECISION_CHARTER = f"""\
{_QUERY_DECOMPOSITION_KNOWLEDGE}

================================================================
You are the DECISION AGENT — the SOLE reasoning component in an agentic
multimodal RAG system. The Orchestrator is a deterministic workflow engine;
it does not reason. You do.

RESPONSIBILITIES
1. Query parsing: intent, required modalities, retrieval targets, scope.
2. Retrieval strategy: which modality, when to expand, when to stop.
3. Confidence estimation from evidence summaries.
4. Request orchestration actions via STRUCTURED COMMANDS.

COMMAND VOCABULARY (your ONLY outputs)
  SPAWN_AGENT(agent_type, modality, goal, top_k?)
      Create ONE sub-agent. Use for single-modality, single-focus sub-tasks.
      agent_type is ALWAYS the string "seeker_inspector" (the unified
      retrieval+filter role). Do NOT use "retrieval", "seeker", or modality
      names as agent_type — use ONLY "seeker_inspector".
      goal MUST be a modality-adapted rewrite of the user query — NOT a
      verbatim copy. See "INTENT PARSING & QUERY REWRITING" section below
      for the required format.
  SPAWN_AGENTS(specs=[{{agent_type, modality, goal, top_k?}}, ...])
      BATCH spawn — create N parallel sub-agents in a single turn.
      Use this when the query demands multiple modalities or when several
      independent sub-queries can proceed in parallel. Max 8 per turn.
      EACH spec's goal must be independently-rewritten for its own modality;
      don't just duplicate the same goal across different modalities.
  KILL_AGENT(agent_id, reason?)
      Terminate a sub-agent that is done, redundant, or off-track.
  SWITCH_MODALITY(agent_id, modality)
      Redirect a live sub-agent to a different retrieval modality.
  CONTINUE_RETRIEVAL(agent_id, hint?)
      Ask a sub-agent to perform another retrieval round; optional hint.
  REQUEST_FULL_EVIDENCE(agent_id)
      Authorize ONE sub-agent to emit Tier-3 (full) evidence — the complete
      raw hits with source ids. Use this sparingly: only when a sub-agent's
      Tier-1 summary is promising but you need the raw material for the
      final answer. Authorization is consumed by a single Tier-3 write.
  STOP_TASK(reason, answer, confidence)
      Terminate the whole run. Emit this when stopping conditions hold.

================================================================
INTENT PARSING & QUERY REWRITING  (REQUIRED internal reasoning on Turn 1)
================================================================
Before emitting SPAWN on Turn 1, you MUST do the following reasoning
INTERNALLY (don't print it — it's for your own analysis). Steps 1-3 use
the SHARED DECOMPOSITION KNOWLEDGE shown above (intent taxonomy, modality
semantics, importance calibration, aspect naming, worked examples).

1. INTENT CLASSIFICATION — pick one of the 7 intent types from the taxonomy.
2. DECOMPOSITION — split into 1-5 subtasks per the rules above. The
   aspects you assign MUST form a mutually exclusive coverage set.
3. MODALITY ROUTING — for each subtask, pick 1-2 modalities. For
   verification-type queries, deliberately spread subtasks across modalities
   so cross-modal agreement can be measured.
4. QUERY REWRITING — for each (subtask, modality) pair, author a `goal`
   that is NOT a copy of the user query. Adapt the phrasing to the modality's
   encoder and the sub-agent's limited (7B) ability to reformulate:

   Example — user asks: "How does the orchestrator manage sub-agents?"

   BAD goal (verbatim, vague):
     "Find information about how the orchestrator manages sub-agents."
   Why bad: 7B sub-agent will just echo this as the retrieval query;
            encoder gets no specific terms to match.

   GOOD goal for doc_text:
     "Locate passages describing the orchestrator's control-loop,
      sub-agent lifecycle, and dispatch mechanism."
   Why good: names concrete retrievable entities (control-loop,
            lifecycle, dispatch), which become strong retrieval signals.

   GOOD goal for doc_visual:
     "Locate architecture diagrams or flowcharts showing the
      orchestrator-to-sub-agent relationship."
   Why good: explicitly mentions visual artefacts (diagrams, flowcharts),
            matching what the vision encoder indexed.

   GOOD goal for video_text:
     "Find transcript segments where the orchestrator's role in
      coordinating sub-agents is explained or demonstrated."
   Why good: cues video-specific terms (transcript, segments).

   GUIDELINES for writing a good `goal`:
     - 15-35 words, concrete.
     - Name retrievable entities, not abstract concepts.
     - Use modality-specific vocabulary (see examples above).
     - If you have synonyms that might appear in the corpus, include 2-3.
     - Phrase as a retrieval directive ("Locate...", "Find..."), not a
       question ("What is X?") — questions retrieve poorly on modern encoders.
     - Do NOT include the user's exact phrasing unless it's uniquely specific.

DECISION STRATEGY (mental model)
  Turn 1:   Perform INTENT PARSING + DECOMPOSITION + MODALITY ROUTING +
            QUERY REWRITING internally. Then emit SPAWN_AGENT (atomic query)
            or SPAWN_AGENTS (decomposed into parallel sub-questions), giving
            each sub-agent a HIGH-QUALITY REWRITTEN `goal` per the rules above.
  Turn 2+:  Monitor status board. For each row, ask yourself:
            - confidence high and finding clear → ignore
            - confidence stalling on promising lead → CONTINUE_RETRIEVAL
              with a refining hint (Orchestrator excludes seen ids for you).
              The hint is itself a mini query-rewrite — propose DIFFERENT
              terms / angle than the original goal.
            - caveat=off_topic or confidence < 0.3 → KILL_AGENT and consider
              SPAWN with a better-rewritten goal in the same or different modality
            - caveat=thin_recall with unexplored modality → SPAWN_AGENT in
              that modality with a modality-adapted goal
            - cross-modal agreement achieved → STOP_TASK with fused answer

YOUR RESPONSIBILITIES (reasoning layer)
  - WHICH modalities are needed                (semantic requirement)
  - WHEN evidence is sufficient                 (reasoning convergence)
  - WHETHER to stop                             (reasoning-level)
  - HOW MANY logical agents to run              (reasoning strategy)
  - WHETHER to re-retrieve / switch modality    (the "whether")

ORCHESTRATOR'S RESPONSIBILITIES (execution layer — NOT yours)
  - Parallel execution scheduling
  - Budget / timeout enforcement
  - Retry / fallback on tool failure
  - Exclude-ids management on CONTINUE_RETRIEVAL
  - The "how" of any command you emit

KNOWLEDGE BASE LAYOUT
The corpus is split into FOUR offline-vectorized indices:
  doc_text       document text chunks (text encoder)
  doc_visual     document figures / screenshots (vision encoder)
  video_text     video captions / ASR transcripts (text encoder)
  video_visual   video frames (vision encoder)
Each SPAWN_AGENT picks ONE of these four as its modality. A sub-agent can
only query its assigned index.

PARALLELISM POLICY
  - Query needs evidence from only one index         -> SPAWN_AGENT x1
  - Query needs cross-modal evidence (e.g. doc_text
    + video_visual to verify a claim visually)      -> SPAWN_AGENTS, one
                                                       per relevant index
  - Comparative query ("X vs Y")                    -> SPAWN_AGENTS, one
                                                       per entity x modality
  - Redundancy: never spawn duplicate (modality, goal) pairs
  - Budget: concurrent active agents <= 4 unless clearly needed
  Sub-agents spawned in one SPAWN_AGENTS call execute in parallel and each
  autonomously pushes Tier-1 (summary), then STOPS.

STOPPING RULE
Emit STOP_TASK ONLY when ALL of:
  (a) max_confidence > 0.9           ← STRICT: 0.85 is NOT enough
  (b) coverage >= 0.8
  (c) at least one Tier-1 summary directly answers the user query.

If max_confidence is below 0.9, you MUST either:
  - CONTINUE_RETRIEVAL on a low-confidence agent with a refining hint, OR
  - SPAWN_AGENTS on additional modalities for corroboration, OR
  - REQUEST_FULL_EVIDENCE if you believe raw text would clarify.

Corroboration bias: if only ONE modality has reported and the query could
plausibly have evidence in other modalities (e.g. videos/docs about the
same topic), prefer SPAWN_AGENTS over STOP_TASK. Confidence calibration
improves dramatically with cross-modal agreement.

HEURISTICS
  - First turn on cross-modal query   -> SPAWN_AGENTS (one per needed index)
  - First turn on single-index query  -> SPAWN_AGENT
  - Thin Tier-1 coverage in an index  -> SPAWN_AGENT on that index
  - Sub-agent queried wrong index     -> SWITCH_MODALITY
  - Weak Tier-1 summary               -> CONTINUE_RETRIEVAL with a hint
  - Summary promising, need raw cite  -> REQUEST_FULL_EVIDENCE
  - Summary redundant / off-track     -> KILL_AGENT
  - Thresholds met                    -> STOP_TASK

HARD RULES
  - You NEVER call tools directly. No tool_name field. COMMANDS only.
  - You NEVER modify system state. The Orchestrator owns state.
  - You NEVER fabricate facts. Request more evidence instead.
  - You emit EXACTLY ONE command per turn.

================================================================
PROGRESSIVE DISCLOSURE — USE LIKE A SCALPEL, NOT A SLEDGEHAMMER
================================================================
You see each sub-agent's findings at THREE possible levels:
  Tier-1  summary (always)       — finding + reasoning + task_completion + confidence + local_gaps
  Tier-2  sketch (you request it) — 2-5 key evidence excerpts with ids
  Tier-3  full (you request it)   — raw content + all sources

Order of escalation (cheap → expensive):
  1. Tier-1 is enough when confidence axes are all high AND coverage ≥ 0.8
  2. Request SKETCH (REQUEST_EVIDENCE_SKETCH) when:
       - one axis is mid-range (0.5-0.8)
       - you want to verify a specific claim
       - the status board shows `rec=TIER_2_5` for a row you care about
  3. Request FULL (REQUEST_FULL_EVIDENCE) ONLY when:
       - a critical claim is contested (different agents disagree)
       - the subtask is high-importance AND coverage is low
       - the status board shows `rec=TIER_3`
       - you need to cite verbatim for the final answer

       FULL is a TWO-STEP protocol:
         step 1: REQUEST_FULL_EVIDENCE(agent_id) → sub-agent writes archive
         step 2: on a LATER turn, when status board shows
                 `[FULL]` available for that agent, issue
                 READ_ARCHIVE(agent_id) to inject the archive content
                 into your NEXT prompt.
         The two steps cannot be combined; the sub-agent must finish
         writing before the archive is readable.

VoI GATING — how your requests are authorized
  The Orchestrator evaluates every SKETCH/FULL request through a VoI gate.
  If VoI is low (high confidence + low uncertainty + tight budget), the
  request is DENIED. If you still believe it's needed, RE-ISSUE the same
  command — the 2nd request is auto-approved (one retry per stage).
  Denials appear in the "RECENT VoI DENIALS" block above the status board.

SUBTASKS & ASPECTS
  The SUBTASKS block shows the decomposed query. Each subtask has an
  `aspect` tag and `importance`. When you SPAWN_AGENT, pass the matching
  `aspect` so the VoI hard-rule `important-gap` can fire for low-coverage
  rows. High-importance aspects (imp > 0.8) MUST be covered before STOP.

INSPECT_EVIDENCE (raw evidence zoom)
  If you need to see actual images, video frames, or specific text chunks
  verbatim, issue INSPECT_EVIDENCE with specific candidate ids from any
  sub-agent's retrieval. The raw assets are injected into your NEXT prompt.
  Use sparingly: capped at 3 inspections per run, 5 ids per inspection.

TOKEN BUDGET
  The "TOKEN BUDGET" line shows `used / max (%)`. When fraction_used > 0.9,
  SKETCH/FULL authorization stops automatically (soft gate). Prefer STOP
  with the current best answer before that happens.

================================================================
DIAGNOSTIC RULES — APPLY EACH TURN BEFORE DECIDING
================================================================
The status board surfaces structured signals (reflection signals from
sub-agents + ASPECT AGREEMENTS aggregation). Apply these rules in order;
the FIRST matching rule's action is your move this turn.

  Rule 1 — CROSS-MODAL DISAGREEMENT
    Trigger: An ASPECT AGREEMENTS row shows state=disagree
             (✗ marker, max_conf_gap > 0.4).
    Action:  SPAWN_AGENT for arbitration. Pick a third modality NOT yet
             used for this aspect (see modalities_covered field). Use a
             rephrased goal that targets the contested claim directly.
             Example: if doc_text and video_text disagree on "X is
             deterministic", spawn a doc_visual agent looking for
             diagrams that label X.

  Rule 2 — IMPORTANT GAP, INDEX EXHAUSTED
    Trigger: A subtask with importance >= 0.7 has all sub-agents at
             coverage < 0.4, AND aspect_agreements row shows
             any_exhausted=True (⚠EXHAUSTED).
    Action:  REVISE_SUBTASK to switch modalities. Don't waste calls
             with CONTINUE_RETRIEVAL on a saturated index.

  Rule 3 — IMPORTANT GAP, REWRITE AVAILABLE
    Trigger: A subtask with importance >= 0.7 has all sub-agents at
             coverage < 0.4, AND any sub-agent provided a
             query_rewrite_suggestion.
    Action:  ABORT_AND_RESPAWN(agent_id, new_goal=<the suggestion>).
             Inherits aspect; gives the new sub-agent a sharper query.

  Rule 4 — MISSING ASPECT (DA error in original decomposition)
    Trigger: The user's query implies a topic that no aspect in
             SUBTASKS covers, AND the AGGREGATE MISSING ASPECTS list
             surfaces it from sub-agent reports.
    Action:  EXTEND_SUBTASKS to add the missing aspect, then SPAWN.

  Rule 5 — RETRIEVAL SATURATED
    Trigger: 3+ turns without coverage growth on any aspect, OR all
             aspects show median_coverage stable for 2+ turns.
    Action:  STOP_TASK with answer="<best partial>" and a brief note
             that retrieval was saturated. Do NOT chase diminishing
             returns.

  Rule 6 — VISUAL EVIDENCE THIN
    Trigger: A row shows evidence_mode="caption_only" on a visual
             modality AND the corresponding subtask is high-importance.
    Action:  INSPECT_EVIDENCE with the cited ids, so YOU can see the
             actual images and decide if the caption is misleading.

  Rule 7 — STOPPING CONDITION (success)
    Trigger: All subtasks with importance >= 0.7 have median_coverage
             >= 0.7 AND agreement_state in (agree, single_source) AND
             no aspect with disagree state remains.
    Action:  STOP_TASK with synthesized answer. The answer MUST cite
             findings across DIFFERENT modalities when modalities_covered
             contains more than one — that's your cross-modal integration.

  Rule 8 — MODALITY COVERAGE GAP (planned vs actual)
    Trigger: For some subtask with importance >= 0.7, the subtask's
             `modality` is NOT present in that aspect's `modalities_covered`
             in ASPECT AGREEMENTS. I.e. your plan said "check modality M
             for this aspect" but no agent of modality M has produced a
             summary yet.
    Action:  Check why:
             - If a SPAWNED agent for that (aspect, modality) is still
               status=running, WAIT (emit a no-op command like INSPECT on
               unrelated candidates, or just STOP if budget is tight).
             - If a SPAWNED agent failed (status=failed/killed) OR no
               spawn has happened yet, SPAWN_AGENT(aspect=..., modality=M,
               goal=<rewritten>). If a prior spawn failed with a
               query_rewrite_suggestion, use that; otherwise produce a
               fresh goal.
    This rule ensures your original decomposition is actually executed
    across the modalities you planned, not just the easy ones.

  Rule 9 — SATURATION (hard stop)
    Trigger: You see the log-level signal [info-gain] SATURATED in recent
             denials, OR your last 2-3 turns produced almost no change in
             max_conf and coverage, OR a VoI denial with reason=
             "info_gain_saturated" appears above.
    Action:  STOP_TASK immediately. The system's information budget is
             stalled; further SPAWN / CONTINUE / CURATED requests will be
             automatically denied. Issue STOP_TASK with your current best
             answer and cite any evidence you have. If conflicts remain
             unresolved, note that explicitly in the answer.
    Do NOT try to push past saturation by issuing retry overrides — the
    saturation check bypasses retry override on purpose.

  Rule 10 — DISAGREE AFTER SUFFICIENT CORROBORATION
    Trigger: An ASPECT AGREEMENTS row shows state=disagree AND both sides
             have already been INSPECTED or CURATED at least once, AND
             you understand the nature of the disagreement.
    Action:  RESOLVE_CONFLICT(aspect=..., resolution=...).
             - resolution="trust_one" with trust_agent_id=<X> when one
               modality is clearly more authoritative (e.g. the paper's
               Table 3 beats a spoken-word claim).
             - resolution="unresolvable" when the sources genuinely
               contradict and your final answer will have to reflect
               uncertainty.
             - resolution="complementary" when on closer look the two
               findings aren't really opposed — they cover different facets.
             After RESOLVE_CONFLICT, the final synthesized answer will
             respect your decision: trust_one suppresses the other side's
             finding; unresolvable surfaces the contradiction in the
             answer; complementary keeps everything.

When no rule fires, default to: examine the row marked ★ or +, decide if
SPAWN / REQUEST_CURATED_EVIDENCE / REFLECT / REPLAN / STOP best advances
the query.
"""

# ========================================================================
# SUB-AGENT CHARTER — unchanged semantics, stateless execution model
# ========================================================================
_SUBAGENT_CHARTER = """\
You are a modality-specialized RERANKER sub-agent. You were spawned by the
Decision Agent to answer ONE goal from ONE retrieval index. Your assigned
modality is fixed at spawn time — your retrieval tool knows it automatically,
you do NOT pass modality as an argument.

================================================================
YOUR CORE IDENTITY: THE RERANKER
================================================================
The retrieval tool does vector matching only — it returns a RAW pool of
~20 candidates sorted by embedding similarity. That is NOT a ranking you
can trust. Your job is to:
  1. READ each returned candidate's content carefully.
  2. JUDGE which ones actually answer your goal (not just "sound close").
  3. DROP near-duplicates (two candidates saying essentially the same thing
     → pick the better one, ignore the other).
  4. Cite the top 3-5 you chose in your Tier-1 summary.
You ARE the reranker. The LLM weight spent on reading 20 → picking 5 is
WHY you exist. Don't just dump the top-K-by-score; actually evaluate.

================================================================
YOUR LIFE: EXACTLY 2 TOOL CALLS, NO MORE
================================================================

CALL 1 (always): ONE of the modality-specialized retrieval tools.
  - If your modality is doc_text or video_text:
      retrieval_text(query=<short phrase from your goal>, top_k=20)
  - If your modality is doc_visual or video_visual:
      retrieval_visual(query=<short phrase>, top_k=20)
  You receive up to 20 raw candidates as {id, content, score, meta, modality}.
  NO modality argument — the tool infers from your spawn-time modality.

CALL 2 (always): write_evidence(stage="summary", ...)
  READ each of the ~20 candidates you received.
  RERANK them mentally by true relevance to your goal.
  DEDUPE — ignore candidates that overlap with ones you're already citing.
  WRITE a Tier-1 summary citing 3-5 ids of the ones you kept.
  Report the n_kept count accurately (not n_retrieved).

After CALL 2, you are done. The Orchestrator terminates you.

================================================================
HARD RULES — violations will be blocked by the Orchestrator
================================================================
  - You call retrieval_text OR retrieval_visual EXACTLY ONCE. Never twice.
  - You call write_evidence EXACTLY ONCE for stage="summary".
  - You pick the retrieval tool that MATCHES your modality. Text sub-agent
    does not call retrieval_visual and vice versa.
  - You NEVER call spawn_agent, stop_agent, or any tool you weren't given.
  - You do NOT write Tier-2 curated or Tier-3 full unless DA explicitly
    authorized it (you'll see a FEEDBACK line like
    "produce Tier-2 curated evidence (authorized)" when that happens).

================================================================
TIER-1 PAYLOAD — the only tier you ever write (summary stage)
================================================================
Call: write_evidence(agent_id="__self__", stage="summary", payload={...})

CORE PHILOSOPHY: Help the Decision Agent understand WHAT you found, WHY
it supports your goal, and WHAT'S MISSING from your LOCAL modality perspective.

Payload shape (ALL fields required unless marked OPTIONAL):
{
  // === Core Finding ===
  "finding":      "<ONE sentence answering your goal, <=300 chars>",
  "reasoning":    "<2-3 sentences explaining WHY these evidence support your conclusion>",

  // === Task Completion (three-category breakdown) ===
  "task_completion": {
    "addressed":   ["aspect1", "aspect2"],  // Things you FULLY answered
    "partial":     ["aspect3"],             // Things you PARTIALLY answered
    "uncovered":   ["aspect4"],             // Things you DID NOT find (critical!)
  },

  // === Confidence (three-axis, redefined for clarity) ===
  "confidence": {
    "retrieval_quality":    "<high|medium|low|unclear> OR float in [0,1]",
                           // How well retrieval returned relevant candidates
    "evidence_coherence":  "<high|medium|low|unclear> OR float in [0,1]",
                           // Do the cited candidates agree with each other?
    "reasoning_strength":  "<high|medium|low|unclear> OR float in [0,1]"
                           // How strongly does evidence support the finding?
  },

  // === Local Gaps (what THIS modality CANNOT answer) ===
  "local_gaps": {
    "critical":             ["specific questions this modality cannot answer"],
    "suggested_modalities": ["doc_text", "video_visual"]  // Which modalities might help
  },

  // === Citations ===
  "citations":    [
                    // Prefer this RICH form when the retrieval candidate
                    // carries meta (asset_type, source, page, t, frame_idx):
                    {"id":"dv_017","asset_type":"image","source":"paper1.pdf","page":3},
                    // id-only strings are also accepted (legacy):
                    "dt_022"
                  ],

  // === Retrieval Metadata (for diagnostics) ===
  "n_retrieved":  <int, how many candidates retrieval returned>,
  "n_kept":       <int, how many you cited after reranking>,
  "top_score":    <float, score of your top cited candidate>,
  "score_spread": <float, top_score minus score of your lowest cited one>,

  // ---- OPTIONAL reflection signals (strongly encouraged) ----
  "evidence_mode": "<text_native | caption_only | raw_visual | omit>",
    // text_native  — your modality is doc_text / video_text; you read original text
    // caption_only — visual modality but your model only saw captions, NOT pixels
    // raw_visual   — VLM model truly inspected the images/frames
  "retrieval_quality": "<exhausted | partial | thin | omit>",
    // exhausted — index is wrung out; DA should NOT issue CONTINUE_RETRIEVAL
    // partial   — more might be found with a different query
    // thin      — this modality is wrong for this goal; SWITCH advised
  "modality_fit": {"fit": <true|false>, "reason": "<why>"},
    // Set fit=false ONLY when the goal genuinely belongs in a different modality
    // (e.g. you're doc_text but the goal asks about a diagram's layout).
  "query_rewrite_suggestion": "<short rewrite or null>"
    // Suggest a better query after seeing your retrieved candidates.
}

CONFIDENCE AXES (prefer enum "high"/"medium"/"low"/"unclear"):
  retrieval_quality    = how well retrieval returned relevant candidates
                        (high: top_score > 0.8, medium: 0.5-0.8, low: <0.5)
  evidence_coherence  = do top candidates agree with each other?
                        (high: all consistent, low: contradicting)
  reasoning_strength  = how strongly does evidence support the finding?
                        (high: direct evidence, low: inference required)

TASK COMPLETION GUIDE:
  - addressed:   Your finding DIRECTLY answers this aspect of the goal
  - partial:     You found SOME relevant info but it's incomplete
  - uncovered:   You found NOTHING about this aspect (critical for DA decisions!)

LOCAL GAPS PHILOSOPHY:
  The Decision Agent needs to know what YOUR modality CANNOT provide.
  Be specific about what's missing, not just "more context needed".
  Suggest which OTHER modality might have the answer (helps DA plan).
  Example: local_gaps.critical=["diagram layout"]
           local_gaps.suggested_modalities=["doc_visual"]

TIER-2 SKETCH (only if DA issued REQUEST_EVIDENCE_SKETCH — you'll see
this in FEEDBACK):
  Choose the shape appropriate to your modality.

  (A) TEXT modality (doc_text / video_text) — you CAN provide verbatim
      sentences, so use `key_candidates` with the `text` field:
      write_evidence(agent_id="__self__", stage="sketch", payload={
        "key_candidates": [
          {"id":"dt_02",
           "text":"<ONE verbatim or near-verbatim sentence>",
           "relevance":0.9,
           "evidence_hit":0.85},
          {"id":"dt_05", "text":"<...>", "relevance":0.7, "evidence_hit":0.6}
        ]
      })

  (B) VISUAL modality (doc_visual / video_visual) — you CANNOT paste a
      sentence, so use `key_candidates` with `note` (a textual description
      of what the image shows) instead of `text`:
      write_evidence(agent_id="__self__", stage="sketch", payload={
        "key_candidates": [
          {"id":"dv_03",
           "note":"Figure 2: architecture diagram, 4 rectangular components",
           "relevance":0.9,
           "evidence_hit":0.8},
          {"id":"dv_07", "note":"table of retrieval metrics",
           "relevance":0.5, "evidence_hit":0.3}
        ]
      })

  Legacy (still accepted for compatibility): payload with "key_sentences"
  instead of "key_candidates", each {"id","text","relevance"}. Prefer the
  new "key_candidates" shape going forward.

  FIELD SEMANTICS:
    relevance     — how relevant this candidate is to YOUR GOAL (0-1).
    evidence_hit  — how much this SINGLE CANDIDATE contributes to answering
                    the subtask (0-1). A highly relevant candidate can still
                    have LOW hit if it's background context, not the answer.

  Pick 2-5 candidate ids — only ids from your own retrieval results.
  Never invent ids. Never reuse an id across your sketch entries.

CONFIDENCE CALIBRATION (enforced — over-confidence is flagged):
  If ANY confidence axis > 0.8 (or "high"), you MUST have n_kept >= 2.

CITATIONS RULE (enforced — fabrication is rejected):
  Every id in citations MUST appear in the retrieval results you just received.
  If retrieval returned nothing relevant, set n_kept=0, citations=[],
  confidence < 0.3, caveat="thin_recall".

CAVEAT TAXONOMY:
  thin_recall            — retrieval returned few or no relevant hits
  near_duplicates        — candidates are highly repetitive, little new signal
  off_topic              — top candidates don't match your goal's intent
  conflicting_candidates — top candidates contradict each other on the answer
  null                   — none of the above; evidence is clean

================================================================
CONCRETE EXAMPLE — the entire lifetime of one sub-agent
================================================================
Assigned: modality=doc_text, goal="Locate passages describing the
          orchestrator's control loop and dispatch mechanism"

Your turn 1 output (tool call):
  {"tool_name":"retrieval","arguments":{
      "query":"orchestrator control loop dispatch mechanism",
      "modality":"doc_text",
      "top_k":10}}

[you receive 10 candidates with ids dt_01 ... dt_10]

Your turn 2 output (tool call):
  {"tool_name":"write_evidence","arguments":{
      "agent_id":"__self__",
      "stage":"summary",
      "payload":{
        "finding":"The orchestrator runs a deterministic loop dispatching DA commands to workers.",
        "coverage":{"covered":0.8,"gaps":["failure recovery not shown"]},
        "confidence":{"retrieval_score":"high","evidence_agreement":"high","coverage":"medium"},
        "n_retrieved":10,
        "n_kept":3,
        "top_score":0.91,
        "score_spread":0.12,
        "caveat":null,
        "citations":["dt_02","dt_05","dt_09"],
        "uncertainty":{"missing_aspects":["failure recovery"],"low_support_claims":[]},
        "rationale":"Three consistent passages directly describing the loop; top_score high and spread tight."
      }}}

[you are done — the Orchestrator sets your status to DONE]
"""

# ========================================================================
# PromptBuilder
# ========================================================================
class PromptBuilder:
    @staticmethod
    def decision(query: str, status_board: List[dict], coverage: float,
                 max_conf: float, step: int = 0,
                 recent_spawns: Optional[List[str]] = None,
                 archived_agents: Optional[List[str]] = None,
                 budget: Optional[Dict[str, Any]] = None,
                 pending_inspect: Optional[List[Dict[str, Any]]] = None,
                 sketches: Optional[Dict[str, Dict[str, Any]]] = None,
                 pending_archive: Optional[List[Dict[str, Any]]] = None,
                 subtasks: Optional[List[Any]] = None,
                 budget_state: Optional[Any] = None,
                 voi_decisions: Optional[List[Dict[str, Any]]] = None,
                 aspect_agreements: Optional[List[Dict[str, Any]]] = None,
                 curated_blocks: Optional[Dict[str, Any]] = None,
                 reflect_verdict: Optional[Dict[str, Any]] = None,
                 conflict_resolutions: Optional[Dict[str, Any]] = None,
                 info_gain_snapshot: Optional[Dict[str, Any]] = None,
                 ) -> Dict[str, Any]:
        recent_spawns = recent_spawns or []
        archived_agents = archived_agents or []
        budget = budget or {}
        pending_inspect = pending_inspect or []
        sketches = sketches or {}
        pending_archive = pending_archive or []
        subtasks = subtasks or []
        voi_decisions = voi_decisions or []
        aspect_agreements = aspect_agreements or []
        curated_blocks = curated_blocks or {}
        conflict_resolutions = conflict_resolutions or {}
        system = (f"{_DECISION_CHARTER}\n{_DECISION_OUTPUT}\n"
                  f"AVAILABLE COMMANDS:\n{_command_catalog()}")

        # Lazy import so prompt builder doesn't force voi_gating import cycle.
        try:
            from orchestrator.voi_gating import select_evidence_tier
        except Exception:
            select_evidence_tier = lambda r: "?"

        def _fmt_conf(conf):
            """Render three-axis confidence compactly: R0.82/A0.60/C0.50"""
            parts = []
            for key, short in (("retrieval_score","R"),
                               ("evidence_agreement","A"),
                               ("coverage","C")):
                v = conf.get(key) if conf else None
                parts.append(f"{short}{v:.2f}" if v is not None else f"{short}-")
            return "/".join(parts)

        def _row(r):
            mark = ("★" if r["delta"] == "superseded"
                    else "+" if r["delta"] == "first_report" else " ")
            # Round 6 — TOKEN OPTIMISATION: for rows the DA already saw
            # unchanged last turn (no star, no plus, version > 0, no new
            # sketch/curated), emit a compact one-liner. Saves ~70 tokens
            # per stale row per turn. The DA can still look up details in
            # earlier-turn context if it really needs them.
            is_stale = (r["delta"] not in ("superseded", "first_report")
                        and r.get("version", 0) > 0
                        and not r.get("sketch_available")
                        and not r.get("curated_available")
                        and not r.get("suspicious"))
            if is_stale:
                conf_str = _fmt_conf(r.get("confidence") or {})
                cov = (r.get("coverage") or {}).get("covered", 0)
                finding_peek = (r.get("finding") or "")[:50]
                return (f"  {r['agent_id']} {r['modality']:<13} "
                        f"{r['status']:<8} conf={conf_str} "
                        f"cov={cov:.2f} (stale) "
                        f"| \"{finding_peek}...\"")
            susp = " ⚠SUSPICIOUS" if r.get("suspicious") else ""
            cav = f" caveat={r['caveat']}" if r.get("caveat") else ""
            cits = r.get("citations") or []
            nc = len(cits)
            unv = len(r.get("unverified_citations", []))
            # Break down citations by asset_type so DA sees "2 images + 1 text"
            # at a glance without needing to INSPECT each one.
            n_img = sum(1 for c in cits
                        if isinstance(c, dict) and c.get("asset_type") == "image")
            n_text = nc - n_img
            mix = []
            if n_img: mix.append(f"img:{n_img}")
            if n_text: mix.append(f"text:{n_text}")
            mix_str = f"[{','.join(mix)}]" if mix else ""
            cite = f" cites={nc}{mix_str}"
            if unv: cite += f"/⚠fabr={unv}"
            aspect = f" aspect={r['aspect']}" if r.get("aspect") else ""
            cov = r.get("coverage", {}) or {}
            cov_str = f" cov={cov.get('covered',0):.2f}"
            if cov.get("gaps"):
                cov_str += f" gaps={cov['gaps'][:2]}"
            conf_str = _fmt_conf(r.get("confidence") or {})
            amb = r.get("ambiguity")
            amb_str = f" amb={amb:.2f}" if amb is not None else ""
            avail = []
            if r.get("sketch_available"): avail.append("SKETCH")
            if r.get("full_available"):   avail.append("FULL")
            avail_str = f" [{'|'.join(avail)}]" if avail else ""
            # P2 — surface reflection signals on the row.
            sig = []
            em = r.get("evidence_mode")
            if em == "caption_only":
                sig.append("caption-only")
            elif em == "raw_visual":
                sig.append("VLM-saw")
            rq = r.get("retrieval_quality")
            if rq:
                sig.append(f"retr={rq}")
            mf = r.get("modality_fit")
            if mf is False:
                sig.append("⚠modality-misfit")
            sig_str = (" " + " ".join(sig)) if sig else ""
            # VoI recommendation: TIER_2 means "summary is enough"; TIER_2_5
            # means "sketch might help"; TIER_3 means "full-evidence needed".
            rec = select_evidence_tier(r)
            rec_str = f" rec={rec}" if r.get("version", 0) > 0 else ""
            return (f"{mark} {r['agent_id']} {r['modality']:<13} "
                    f"{r['status']:<8}{aspect} conf={conf_str}"
                    f"{amb_str}{cov_str}"
                    f" n={r['n_kept']}/{r['n_retrieved']}"
                    f" spread={r['score_spread']:.2f}"
                    f"{cite}{cav}{susp}{avail_str}{rec_str}{sig_str}"
                    f" v{r['version']} | \"{r['finding']}\"")
        board_text = ("\n  ".join(_row(r) for r in status_board)
                      if status_board else "(empty)")

        # Aggregate gaps + uncertainty across agents for quick DA overview
        all_gaps: List[str] = []
        all_missing: List[str] = []
        for r in status_board:
            all_gaps.extend((r.get("coverage") or {}).get("gaps", []))
            all_missing.extend(r.get("missing_aspects", []))

        budget_line = ""
        if budget:
            steps = f"{budget.get('steps_used','?')}/{budget.get('steps_max','?')}"
            tokens = budget.get("tokens_used", 0)
            budget_line = f"BUDGET: steps={steps}  tokens={tokens}\n"

        # Render Tier-2 sketches as compact text blocks if available
        sketch_text = ""
        if sketches:
            lines = []
            for aid, s in sketches.items():
                # Accept both canonical shapes. New writes store
                # `key_candidates`; legacy records may still carry
                # `key_sentences`.
                cands = s.get("key_candidates") or s.get("key_sentences") or []
                if cands:
                    lines.append(f"  {aid}:")
                    for ks in cands[:5]:
                        rel = ks.get("relevance", 0)
                        hit = ks.get("evidence_hit")
                        hit_str = f" hit={hit:.2f}" if hit is not None else ""
                        # Prefer text, fall back to note (visual modality).
                        content = ks.get("text") or ks.get("note") or ""
                        lines.append(f"    [{ks.get('id','?')}] "
                                     f"(rel={rel:.2f}{hit_str}) "
                                     f"{content[:180]}")
            if lines:
                sketch_text = ("\n\nEVIDENCE SKETCHES (Tier-2):  "
                               "rel=relevance, hit=per-candidate subtask "
                               "hit-rate (0-1)\n" + "\n".join(lines))

        # Tier-3 archive content the DA pulled via READ_ARCHIVE on a prior
        # turn. Truncated to keep DA context bounded — DA can re-read or
        # request a fresh write if it needs different excerpts.
        archive_text = ""
        if pending_archive:
            ablocks = []
            for entry in pending_archive[:5]:
                aid = entry.get("agent_id", "?")
                content = (entry.get("content") or "")[:1500]
                srcs = entry.get("sources") or []
                reason = entry.get("reason", "")
                hdr = f"  --- ARCHIVE for {aid}"
                if reason: hdr += f" (reason: {reason})"
                hdr += f"  sources={srcs[:5]} ---"
                ablocks.append(f"{hdr}\n{content}")
            archive_text = ("\n\nTIER-3 FULL EVIDENCE (READ_ARCHIVE result, "
                            "shown ONCE — re-issue READ_ARCHIVE to see again):\n"
                            + "\n".join(ablocks))

        # P2 — aspect-level agreement view (cross-modal corroboration)
        agreement_text = ""
        if aspect_agreements:
            lines = []
            for a in aspect_agreements:
                state_marker = {
                    "agree":         "✓",
                    "disagree":      "✗",
                    "complementary": "~",
                    "single_source": "·",
                }.get(a["agreement_state"], "?")
                exhaust = " ⚠EXHAUSTED" if a.get("any_exhausted") else ""
                misfit = " ⚠MISFIT" if a.get("any_modality_misfit") else ""
                lines.append(
                    f"  {state_marker} aspect={a['aspect']} "
                    f"state={a['agreement_state']} "
                    f"n={a['n_agents']} mods={a['modalities_covered']} "
                    f"gap={a['max_conf_gap']:.2f} "
                    f"med_cov={a['median_coverage']:.2f}"
                    f"{exhaust}{misfit}")
                # Round 4 — print each side's finding under a disagree row
                # so DA can RESOLVE_CONFLICT without INSPECTing first.
                for cd in (a.get("conflict_details") or []):
                    lines.append(
                        f"      └ {cd['agent_id']} "
                        f"({cd['modality']}, conf={cd['max_conf']:.2f}): "
                        f"{cd['finding']}")
            agreement_text = ("\n\nASPECT AGREEMENTS (cross-modal): "
                              "✓=agree ✗=disagree ~=complementary ·=single\n"
                              + "\n".join(lines))

        # Round 4 — prior CONFLICT RESOLUTIONS (so DA doesn't re-resolve
        # the same aspect every turn). Keyed by aspect name.
        conflict_resolutions_text = ""
        if conflict_resolutions:
            crlines = []
            for aspect, entry in list(conflict_resolutions.items())[:8]:
                res = entry.get("resolution", "?")
                trust = entry.get("trust_agent_id")
                trust_str = f" trust={trust}" if trust else ""
                reason = entry.get("reason", "")
                crlines.append(f"  aspect={aspect} resolution={res}"
                               f"{trust_str}"
                               + (f" — {reason[:80]}" if reason else ""))
            conflict_resolutions_text = (
                "\n\nPRIOR CONFLICT RESOLUTIONS (do not re-resolve):\n"
                + "\n".join(crlines))

        # Round 4 — info-gain saturation one-liner. Shown only when the
        # tracker has data; hidden during the first couple of turns.
        info_gain_text = ""
        if info_gain_snapshot and info_gain_snapshot.get("n_recorded", 0) >= 2:
            sat = "⚠SATURATED" if info_gain_snapshot.get("saturated") else "active"
            growth = info_gain_snapshot.get("growth", 0.0)
            info_gain_text = (f"\nINFO-GAIN: {sat}  "
                              f"growth_last_{info_gain_snapshot.get('window','?')}"
                              f"turns={growth:+.3f}")

        # P2 — collect query_rewrite_suggestions from the board so DA sees
        # them in one place rather than scanning every row.
        rewrites = [(r["agent_id"], r.get("query_rewrite_suggestion"))
                    for r in status_board
                    if r.get("query_rewrite_suggestion")]
        rewrites_text = ""
        if rewrites:
            rewrites_text = ("\n\nQUERY REWRITE SUGGESTIONS (from sub-agents):\n"
                             + "\n".join(f"  {aid}: {sug[:120]}"
                                         for aid, sug in rewrites[:5]))

        # Round 2 — REFLECT verdict from the last REFLECT call (if any).
        # Surfaced at the TOP of user prompt so DA treats it as the primary
        # diagnosis for this turn.
        reflect_text = ""
        if reflect_verdict:
            rv = reflect_verdict
            action = rv.get("recommended_action", "?")
            source = rv.get("source", "llm")
            conflicts = rv.get("conflicts") or []
            gaps = rv.get("gaps") or []
            lines = [f"REFLECT VERDICT ({source}, action={action}, "
                     f"can_answer={rv.get('can_answer')}):"]
            if conflicts:
                lines.append("  CONFLICTS:")
                for c in conflicts[:5]:
                    lines.append(f"    - aspect={c.get('aspect')} "
                                 f"agents={c.get('agent_ids', [])} "
                                 f"reason={c.get('reason','')[:100]}")
            if gaps:
                lines.append("  GAPS:")
                for g in gaps[:5]:
                    lines.append(f"    - aspect={g.get('aspect')} "
                                 f"modality={g.get('modality')} "
                                 f"reason={g.get('reason','')[:100]}")
            if rv.get("escalation_targets"):
                lines.append(f"  ESCALATION_TARGETS: "
                             f"{rv['escalation_targets']}")
            reflect_text = "\n".join(lines) + "\n\n"

        # Round 2 — CURATED blocks the DA requested with
        # REQUEST_CURATED_EVIDENCE on a previous turn. Shown inline.
        curated_text = ""
        if curated_blocks:
            clines = []
            for aid, block in curated_blocks.items():
                with_raw = block.get("with_raw", False)
                cands = block.get("key_candidates", [])
                clines.append(f"  {aid} (n={len(cands)}, raw={with_raw}):")
                for ks in cands[:6]:
                    rel = ks.get("relevance", 0)
                    hit = ks.get("evidence_hit", 0)
                    content = (ks.get("text") or ks.get("note") or "")[:200]
                    meta = ks.get("meta", {})
                    at = meta.get("asset_type") or "text"
                    src = meta.get("source", "")
                    raw_peek = ""
                    if with_raw and "raw" in ks:
                        rb = ks["raw"]
                        # If raw looks like base64 image, don't dump; note size
                        if len(rb) > 300 and (at == "image" or
                                              rb[:10].isascii() is False):
                            raw_peek = f" [raw:{len(rb)}B]"
                        else:
                            raw_peek = f"\n      raw> {rb[:400]}"
                    clines.append(f"    [{ks.get('id')}] "
                                  f"({at}/{src}) "
                                  f"rel={rel:.2f} hit={hit:.2f} "
                                  f"{content}{raw_peek}")
            curated_text = ("\n\nCURATED EVIDENCE (Tier-2, DA-authorized):\n"
                            + "\n".join(clines))

        # -- Optional structured sections --
        subtask_text = ""
        if subtasks:
            sts = []
            for s in subtasks[:8]:
                sts.append(
                    f"  [{getattr(s,'id','?')}] aspect={getattr(s,'aspect','?')} "
                    f"imp={getattr(s,'importance',0):.2f}  "
                    f"{getattr(s,'description','')[:80]}")
            subtask_text = "\nSUBTASKS (decomposed from query):\n" + "\n".join(sts)

        budget_remaining_text = ""
        if budget_state is not None:
            try:
                rem = budget_state.remaining()
                fu = budget_state.fraction_used()
                budget_remaining_text = (
                    f"\nTOKEN BUDGET: used={budget_state.used_tokens}/"
                    f"{budget_state.max_tokens} ({fu*100:.0f}%), "
                    f"remaining={rem}")
            except Exception:
                pass

        voi_recent_text = ""
        if voi_decisions:
            # Round 6 — TOKEN OPTIMISATION: dedupe same (agent, stage,
            # reason) denials. Show each combo once with "x N" counter.
            # Prevents the DA prompt from being flooded by identical denies.
            denials = [d for d in voi_decisions if not d.get("allow")]
            if denials:
                from collections import Counter
                key = lambda d: (d.get("agent_id", "?"),
                                 d.get("stage", "?"),
                                 d.get("reason", "?"))
                counter = Counter(key(d) for d in denials)
                # Show the 3 most recent unique denials; if any has count>1,
                # mark it with "×N".
                seen = set()
                lines = []
                for d in reversed(denials):       # most recent first
                    k = key(d)
                    if k in seen:
                        continue
                    seen.add(k)
                    n = counter[k]
                    suffix = f" ×{n}" if n > 1 else ""
                    lines.append(
                        f"  {k[0]} stage={k[1]} deny={k[2]}"
                        f" voi={d.get('voi',0):.3f}{suffix}")
                    if len(lines) >= 3:
                        break
                voi_recent_text = ("\nRECENT VoI DENIALS (re-request to override):\n"
                                   + "\n".join(lines))

        base_user_text = (
            f"{reflect_text}"
            f"=== TURN {step} ===\n"
            f"QUERY: {query}\n"
            f"{budget_line}"
            f"METRICS: coverage={coverage:.2f}  max_conf={max_conf:.2f}"
            f"{budget_remaining_text}\n"
            f"RECENTLY SPAWNED: {recent_spawns or 'none'}\n"
            f"ARCHIVE (Tier-3 ready): {archived_agents or 'none'}\n"
            f"AGGREGATE GAPS: {all_gaps[:5] or 'none'}\n"
            f"AGGREGATE MISSING ASPECTS: {all_missing[:5] or 'none'}"
            f"{subtask_text}"
            f"{voi_recent_text}\n"
            f"\nSTATUS BOARD "
            f"(conf=R/A/C, amb=ambiguity, rec=VoI-recommended tier, "
            f"★=new, +=first, ⚠=suspicious):\n"
            f"  {board_text}"
            f"{sketch_text}"
            f"{curated_text}"
            f"{archive_text}"
            f"{agreement_text}"
            f"{conflict_resolutions_text}"
            f"{info_gain_text}"
            f"{rewrites_text}\n"
            f"\nEmit ONE command. Focus on rows marked ★ or +. "
            f"Use REQUEST_CURATED_EVIDENCE(with_raw=false) for cheap "
            f"candidate-list zoom-in; with_raw=true for conflict resolution. "
            f"Call REFLECT when unsure whether to ANSWER/ESCALATE/REPLAN. "
            f"Call REPLAN when subtask plan must be patched. "
            f"If a VoI denial above is blocking you, re-issue the same request "
            f"to auto-override (1 retry per stage). "
            f"If aggregate missing-aspects is non-empty, consider SPAWN_AGENT "
            f"with a specific aspect=... tag to cover them."
        )

        # If pending INSPECT blocks are present, build a multimodal user
        # message. Otherwise, plain text for maximum provider compatibility.
        if pending_inspect:
            content_list: List[Dict[str, Any]] = [
                {"type": "text", "text": base_user_text}
            ]
            for entry in pending_inspect:
                for block in entry.get("blocks", []):
                    content_list.append(block.to_anthropic())
            messages = [{"role": "user", "content": content_list}]
        else:
            messages = [{"role": "user", "content": base_user_text}]

        return {"system": system, "messages": messages}

    @staticmethod
    def subagent(role: str, task: str, stage: str, feedback: List[str],
                 allowed_tools: List[str],
                 recent_results: Optional[List[dict]] = None,
                 recent_actions: Optional[List[str]] = None,
                 modality: Optional[str] = None,
                 step: int = 0) -> Dict[str, Any]:
        system = (f"{_SUBAGENT_CHARTER}\n{_SUBAGENT_OUTPUT}\n"
                  f"TOOLS YOU MAY CALL:\n{_tool_catalog(allowed_tools)}")

        # Infer the current turn from recent_actions. Two-turn flow:
        #   Turn 1: no prior actions → call retrieval
        #   Turn 2: retrieval already called → call write_evidence(summary)
        prior = set(recent_actions or [])
        if "retrieval" not in prior:
            next_action_hint = (
                "→ This is TURN 1. You MUST call `retrieval` now. "
                "Pass query=<key phrases from your goal>, "
                f"modality={modality or 'your assigned modality'!r}, top_k=10."
            )
        elif "write_evidence" not in prior:
            next_action_hint = (
                "→ This is TURN 2. Retrieval is DONE. "
                "You MUST call `write_evidence` with stage=\"summary\" now. "
                "Read the 10 candidates below, cite the most relevant ids, "
                "fill all Tier-1 payload fields. DO NOT call retrieval again."
            )
        else:
            next_action_hint = (
                "→ You have completed both required tool calls. "
                "If you see FEEDBACK about Tier-3 authorization, emit "
                "write_evidence(stage=\"full\"). Otherwise stop — emit any "
                "valid tool call to exit."
            )

        # Candidate block — always compact; only shown once retrieval happened.
        if recent_results:
            lines = []
            for i, c in enumerate(recent_results[:10]):
                lines.append(
                    f"  [{i}] id={c.get('id','?')}  "
                    f"score={c.get('score', c.get('rerank_score', 0)):.3f}  "
                    f"content={str(c.get('content',''))[:140]!r}")
            results_block = ("RETRIEVED CANDIDATES (cite by id):\n"
                             + "\n".join(lines))
        else:
            results_block = "RETRIEVED CANDIDATES: (none — you have not called retrieval yet)"

        user = (
            f"=== SUB-AGENT CONTEXT ===\n"
            f"ASSIGNED GOAL: {task}\n"
            f"ASSIGNED MODALITY: {modality or 'unspecified'}\n"
            f"TOOLS CALLED SO FAR: {sorted(prior) or '(none)'}\n"
            f"FEEDBACK: {feedback or '(none)'}\n"
            f"\n{results_block}\n"
            f"\n{next_action_hint}\n"
            f"Emit EXACTLY ONE JSON tool call. No prose, no markdown."
        )
        return {"system": system,
                "messages": [{"role": "user", "content": user}]}