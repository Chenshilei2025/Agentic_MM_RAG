"""Evidence pool — three-layer model tuned for Decision-Agent attention.

LAYER A  Scratch        — sub-agent private working area (candidates, rerank
                          intermediates). DA never sees it.
LAYER B  Status Board   — one AgentSnapshot per sub-agent, SUPERSEDE on write.
                          This is the DA's default view.
LAYER C  Archive        — raw tier-3 evidence keyed by agent_id. DA queries
                          this on demand during final answer synthesis.

Intent entries are retained in the Orchestrator's trace for debugging but are
NOT shown to the Decision Agent — they add noise without improving decisions.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set
import threading, time, uuid

# -- Tier schemas --
# Tier 2 = summary. Default view for the Decision Agent. Structured so the DA
# can decide "what's covered / where are the gaps / how confident along which
# axis" without seeing raw evidence.
_TIER2_REQUIRED = {"finding", "coverage", "confidence", "n_retrieved",
                   "n_kept", "score_spread", "citations"}
# Tier 2.5 = evidence sketch. Returned ONLY when DA issues
# REQUEST_EVIDENCE_SKETCH. Sits between tier-2 (lossy summary) and tier-3
# (full content). Designed for progressive evidence zooming: DA can inspect
# the most relevant 2-5 quoted sentences without paying the full cost.
# Tier 2.5 = evidence sketch. Accepts either shape:
#   {"key_sentences": [{"id","text","relevance"}]}    — legacy, text-only
#   {"key_candidates": [{"id","relevance","evidence_hit",
#                        "note"?, "text"?}]}          — new, works for text + visual
# Exactly ONE of the two keys must be present.
_TIER25_ACCEPTED = {"key_sentences", "key_candidates"}
# Tier 3 = full content. Only after REQUEST_FULL_EVIDENCE authorization.
_TIER3_REQUIRED = {"content", "sources"}
# Tier 1 = intent. Orchestrator auto-writes it at spawn time.
_TIER1_REQUIRED = {"modality", "data_source", "planned_k"}

CAVEAT_ENUM = {None, "thin_recall", "near_duplicates", "off_topic",
               "conflicting_candidates"}

# P2 — reflection-signal enums
EVIDENCE_MODE_ENUM = {None, "raw_visual", "caption_only", "text_native"}
RETRIEVAL_QUALITY_ENUM = {None, "exhausted", "partial", "thin"}

# Sub-agent uses coarse enums for confidence axes; 7B models estimate
# discrete buckets more reliably than decimals. Orchestrator maps to floats.
_CONF_AXIS_ENUM = {"high", "medium", "low", "unclear", None}
_CONF_ENUM_TO_FLOAT = {"high": 0.85, "medium": 0.55, "low": 0.25,
                      "unclear": None, None: None}


class DisclosureError(Exception):
    """Raised on tier schema violation or unauthorized tier-3 write."""


@dataclass
class AgentSnapshot:
    """One row of the DA's Status Board — the current latest report
    from a sub-agent. Overwritten on each new tier-2 write.

    Schema upgrade for progressive disclosure:
      - coverage       : which aspect this agent tackled + how much of it
      - confidence     : three-axis decomposition (retrieval / agreement / coverage)
      - uncertainty    : explicit missing-aspects and low-support-claims
      - sketch         : Tier-2.5 key sentences, populated on authorized request
      - full_available : whether Tier-3 has been written
    """
    agent_id: str
    modality: str
    goal: str
    aspect: Optional[str] = None               # DA-assigned aspect tag
    status: str = "running"
    finding: str = ""
    # NEW — coverage of the assigned aspect
    coverage_covered: float = 0.0              # [0,1] how well aspect is answered
    coverage_gaps: List[str] = field(default_factory=list)
    # NEW — three-axis confidence (floats, mapped from sub-agent's enum)
    conf_retrieval: Optional[float] = None     # objective, from retrieval score
    conf_agreement: Optional[float] = None     # candidate-level agreement
    conf_coverage: Optional[float] = None      # alignment with the goal
    # NEW — ambiguity signal used by VoI gating. Either reported by the
    # sub-agent (if it estimated) or derived from score_spread (fallback).
    ambiguity: Optional[float] = None
    # Retained — retrieval diagnostics
    n_retrieved: int = 0
    n_kept: int = 0
    top_score: float = 0.0
    score_spread: float = 0.0
    caveat: Optional[str] = None
    rationale: str = ""
    suspicious_confidence: bool = False
    # Uncertainty — explicit language, not just numbers
    missing_aspects: List[str] = field(default_factory=list)
    low_support_claims: List[str] = field(default_factory=list)
    # Citations
    # Citations — each entry is a dict {id, asset_type, source, page, t, ...}
    # (P2 round-2). Unverified citations (ids not seen in retrieval) stay as
    # plain id strings so DA can still spot fabrication.
    citations: List[Dict[str, Any]] = field(default_factory=list)
    unverified_citations: List[str] = field(default_factory=list)
    # P2 — reflection signals from the sub-agent. All optional; sub-agent
    # may omit any of them. They give the DA structured hints about WHY a
    # sub-agent reached its current state, so DA can pick the right repair
    # action (REVISE / SWITCH / ABORT / etc.) rather than guessing.
    evidence_mode: Optional[str] = None
        # "raw_visual"  — VLM truly saw the image/frame
        # "caption_only" — only had the caption text (limit-of-what-it-saw)
        # "text_native"  — text modality, content IS the evidence
    retrieval_quality: Optional[str] = None
        # "exhausted" — this query+index is wrung out; CONTINUE is wasted
        # "partial"   — more available, needs query rewrite
        # "thin"      — this modality wrong for this goal; SWITCH advised
    modality_fit: Optional[bool] = None
    modality_fit_reason: str = ""
    query_rewrite_suggestion: Optional[str] = None
    # Progressive disclosure slots
    sketch: Optional[Dict[str, Any]] = None    # Tier-2.5 key_sentences
    sketch_authorized: bool = False            # has DA issued REQUEST_EVIDENCE_SKETCH?
    # Round 2 — unified curated disclosure. `curated` holds a summary view
    # {n_candidates, with_raw}; full payload is in pool._curated[agent_id].
    curated: Optional[Dict[str, Any]] = None
    full_available: bool = False               # has Tier-3 been written?
    # Versioning
    version: int = 0
    last_updated: float = field(default_factory=time.time)
    delta: str = "first_report"


class EvidencePool:
    """Three-layer store. Thread-safe. Supersede semantics on the board."""

    def __init__(self, bypass_tier3_gate: bool = False,
                 finding_max_chars: int = 200):
        self._lock = threading.Lock()
        self._board: Dict[str, AgentSnapshot] = {}
        self._archive: Dict[str, List[Dict[str, Any]]] = {}
        self._sketches: Dict[str, Dict[str, Any]] = {}   # agent_id → sketch
        self._trace: List[Dict[str, Any]] = []
        self._full_authorized: Set[str] = set()
        self._sketch_authorized: Set[str] = set()
        # Round 2 — curated authorizations. Two tiers:
        #   light: agent_id → True      (summary-level curated list, no raw)
        #   raw:   agent_id → set(ids)  (raw blobs for specific ids; all-ids
        #                                marker = True means "all top-K")
        self._curated_light_authorized: Set[str] = set()
        self._curated_raw_authorized: Dict[str, Any] = {}
        self._curated: Dict[str, Dict[str, Any]] = {}   # agent_id → payload
        self._known_ids: Dict[str, Set[str]] = {}
        # P2 round-2: per-agent map id → meta dict, populated by
        # note_retrieved_candidates(). Used to structure citations (attach
        # asset_type/source/page/t) even when the sub-agent only returned ids.
        self._known_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._bypass_tier3_gate = bypass_tier3_gate
        self._finding_max_chars = finding_max_chars
        if bypass_tier3_gate:
            import warnings
            warnings.warn(
                "EvidencePool tier-3 gate DISABLED (ablation mode).",
                UserWarning, stacklevel=2)

    # ---------------- Agent registration (by Orchestrator only) ----------
    def register_agent(self, agent_id: str, modality: str, goal: str,
                       aspect: Optional[str] = None):
        with self._lock:
            if agent_id not in self._board:
                self._board[agent_id] = AgentSnapshot(
                    agent_id=agent_id, modality=modality,
                    goal=goal, aspect=aspect)
            self._known_ids.setdefault(agent_id, set())

    def note_retrieved_ids(self, agent_id: str, ids: List[str]) -> None:
        with self._lock:
            self._known_ids.setdefault(agent_id, set()).update(ids)

    def note_retrieved_candidates(self, agent_id: str,
                                  candidates: List[Dict[str, Any]]) -> None:
        """Register full candidates so downstream code can enrich citations
        with their asset_type/source/page/t — letting DA know WHAT kind of
        evidence a cited id points to, without DA having to INSPECT first.

        Each candidate is {id, content, score, meta}. We store a flattened
        extract: {id: {asset_type, source, page, t, frame_idx, ...}}.
        """
        with self._lock:
            known_ids = self._known_ids.setdefault(agent_id, set())
            known_meta = self._known_meta.setdefault(agent_id, {})
            for c in candidates:
                cid = str(c.get("id", ""))
                if not cid:
                    continue
                known_ids.add(cid)
                meta = c.get("meta") or {}
                # Project the fields DA cares about. Unknown keys preserved.
                known_meta[cid] = {
                    "asset_type": meta.get("asset_type"),  # "image" | None
                    "source":     meta.get("source"),
                    "page":       meta.get("page"),
                    "t":          meta.get("t") or meta.get("t_start"),
                    "frame_idx":  meta.get("frame_idx"),
                    "modality":   c.get("modality"),
                }

    def set_agent_status(self, agent_id: str, status: str):
        with self._lock:
            snap = self._board.get(agent_id)
            if snap: snap.status = status

    # ---------------- Authorization (DA-initiated, Orchestrator-executed) --
    def authorize_sketch(self, agent_id: str) -> None:
        """Grant one-time Tier-2.5 write permission."""
        with self._lock:
            self._sketch_authorized.add(agent_id)
            snap = self._board.get(agent_id)
            if snap: snap.sketch_authorized = True

    def is_authorized_for_sketch(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._sketch_authorized

    # ---------------- tier-3 gate ---------------------------------------
    def authorize_full(self, agent_id: str) -> None:
        with self._lock:
            self._full_authorized.add(agent_id)

    def is_authorized_for_full(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._full_authorized

    # ---------------- curated gate (Round 2, unified SKETCH+INSPECT) ------
    def authorize_curated(self, agent_id: str,
                          with_raw: bool = False,
                          ids: Optional[List[str]] = None) -> None:
        """DA says: this sub-agent may write a `curated` block.
        with_raw=True authorises inclusion of raw blobs for matching ids
        (or all top-K if ids is None)."""
        with self._lock:
            self._curated_light_authorized.add(agent_id)
            if with_raw:
                # Store ids set (or True marker for "all top-K")
                self._curated_raw_authorized[agent_id] = (
                    set(ids) if ids else True)

    def is_authorized_curated(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._curated_light_authorized

    def get_curated(self, agent_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._curated.get(agent_id)

    # ---------------- write paths (called by tools) ---------------------
    def write(self, agent_id: str, stage: str,
              payload: Dict[str, Any], aspect_agreements: Optional[List[Dict[str, Any]]] = None) -> str:
        if stage == "intent":
            return self._write_intent(agent_id, payload)
        if stage == "summary":
            return self._write_summary(agent_id, payload)
        if stage == "sketch":
            return self._write_sketch(agent_id, payload)
        if stage == "curated":
            return self._write_curated(agent_id, payload, aspect_agreements)
        if stage == "full":
            return self._write_full(agent_id, payload)
        raise DisclosureError(f"invalid stage: {stage}")

    def _write_intent(self, agent_id, payload):
        missing = _TIER1_REQUIRED - set(payload.keys())
        if missing:
            raise DisclosureError(
                f"intent missing fields: {sorted(missing)}")
        eid = str(uuid.uuid4())
        with self._lock:
            self._trace.append({"id": eid, "agent_id": agent_id,
                                "stage": "intent", "payload": dict(payload),
                                "ts": time.time()})
        return eid

    def _write_summary(self, agent_id, payload):
        missing = _TIER2_REQUIRED - set(payload.keys())
        if missing:
            raise DisclosureError(
                f"summary missing fields: {sorted(missing)}")
        finding = str(payload["finding"])[:self._finding_max_chars]
        caveat = payload.get("caveat")
        if caveat not in CAVEAT_ENUM:
            raise DisclosureError(
                f"invalid caveat: {caveat!r}; allowed: "
                f"{sorted(x for x in CAVEAT_ENUM if x)}")

        # --- Coverage block ---
        coverage = payload["coverage"]
        if not isinstance(coverage, dict):
            raise DisclosureError("coverage must be an object")
        cov_covered = float(coverage.get("covered", 0.0))
        if not 0.0 <= cov_covered <= 1.0:
            raise DisclosureError(
                f"coverage.covered must be in [0,1], got {cov_covered}")
        cov_gaps = coverage.get("gaps", [])
        if not isinstance(cov_gaps, list):
            raise DisclosureError("coverage.gaps must be a list of strings")
        cov_gaps = [str(g)[:200] for g in cov_gaps[:6]]

        # --- Confidence three axes (enum or float) ---
        conf = payload["confidence"]
        if not isinstance(conf, dict):
            raise DisclosureError("confidence must be an object with "
                                  "retrieval/agreement/coverage axes")
        def _axis(name, required=True):
            v = conf.get(name)
            if v is None:
                if required:
                    raise DisclosureError(
                        f"confidence.{name} is required")
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                if not 0.0 <= f <= 1.0:
                    raise DisclosureError(
                        f"confidence.{name} float must be in [0,1]")
                return f
            if isinstance(v, str):
                if v not in _CONF_AXIS_ENUM:
                    raise DisclosureError(
                        f"confidence.{name} enum must be one of "
                        f"{sorted(x for x in _CONF_AXIS_ENUM if x)}")
                return _CONF_ENUM_TO_FLOAT[v]
            raise DisclosureError(
                f"confidence.{name} must be number or enum string")
        conf_retrieval = _axis("retrieval_score")
        conf_agreement = _axis("evidence_agreement")
        conf_coverage  = _axis("coverage")

        n_kept = int(payload["n_kept"])

        # --- Citation grounding ---
        # Citations may be id-only strings (legacy) OR structured dicts
        # {"id", "asset_type"?, "source"?, "page"?, "t"?, "frame_idx"?}
        # (new). Either way we partition into verified vs fabricated and
        # always enrich verified entries with known_meta from retrieval.
        raw_cites = payload.get("citations", [])
        if not isinstance(raw_cites, list):
            raise DisclosureError("citations must be a list")
        with self._lock:
            known = self._known_ids.get(agent_id, set())
            known_meta = dict(self._known_meta.get(agent_id, {}))

        cites: List[Dict[str, Any]] = []
        unverified: List[str] = []
        for entry in raw_cites:
            if isinstance(entry, str):
                cid = entry
                user_meta: Dict[str, Any] = {}
            elif isinstance(entry, dict):
                cid = str(entry.get("id", ""))
                user_meta = {k: v for k, v in entry.items() if k != "id"}
            else:
                continue
            if not cid:
                continue
            if cid not in known:
                unverified.append(cid)
                continue
            # Merge: start with known_meta, then override with any fields
            # the sub-agent explicitly provided.
            merged = dict(known_meta.get(cid, {}))
            merged.update({k: v for k, v in user_meta.items() if v is not None})
            merged["id"] = cid
            cites.append(merged)

        if n_kept > 0 and not cites:
            raise DisclosureError(
                f"tier-2 from {agent_id} has no verifiable citations; "
                f"provided={[str(c)[:40] for c in raw_cites]}, "
                f"known retrieval ids={sorted(known)[:10]}")

        # Max confidence across axes used for suspicious-confidence gate.
        axis_floats = [x for x in (conf_retrieval, conf_agreement,
                                   conf_coverage) if x is not None]
        max_conf = max(axis_floats) if axis_floats else 0.0
        suspicious = (max_conf > 0.8 and
                      (n_kept < 2 or caveat is not None
                       or len(unverified) > 0))

        # --- Uncertainty block (optional) ---
        unc = payload.get("uncertainty") or {}
        missing_aspects = [str(x)[:200] for x in unc.get("missing_aspects", [])][:6]
        low_support = [str(x)[:200] for x in unc.get("low_support_claims", [])][:6]

        # --- Ambiguity (option C: use sub-agent's estimate if present,
        #     else derive from score_spread). Rationale: spread small means
        #     candidates agree → low ambiguity; large spread means candidates
        #     disagree → high ambiguity. Bounded to [0,1]. ---
        raw_amb = payload.get("ambiguity")
        if isinstance(raw_amb, (int, float)):
            ambiguity = max(0.0, min(1.0, float(raw_amb)))
        else:
            spread = float(payload["score_spread"])
            # Spread in [0, ~1]. If n_kept < 2, ambiguity is undefined; default 0.5.
            ambiguity = (min(1.0, spread * 2.0)
                         if n_kept >= 2 else 0.5)

        # --- P2 reflection signals (all optional, all validated) ---
        evidence_mode = payload.get("evidence_mode")
        if evidence_mode not in EVIDENCE_MODE_ENUM:
            raise DisclosureError(
                f"invalid evidence_mode: {evidence_mode!r}; allowed: "
                f"{sorted(x for x in EVIDENCE_MODE_ENUM if x)}")
        retrieval_quality = payload.get("retrieval_quality")
        if retrieval_quality not in RETRIEVAL_QUALITY_ENUM:
            raise DisclosureError(
                f"invalid retrieval_quality: {retrieval_quality!r}; allowed: "
                f"{sorted(x for x in RETRIEVAL_QUALITY_ENUM if x)}")
        mod_fit_obj = payload.get("modality_fit")
        modality_fit_val: Optional[bool] = None
        modality_fit_reason = ""
        if isinstance(mod_fit_obj, dict):
            modality_fit_val = bool(mod_fit_obj.get("fit", True))
            modality_fit_reason = str(mod_fit_obj.get("reason", ""))[:200]
        elif isinstance(mod_fit_obj, bool):
            modality_fit_val = mod_fit_obj
        query_rewrite_suggestion = payload.get("query_rewrite_suggestion")
        if query_rewrite_suggestion is not None:
            query_rewrite_suggestion = str(query_rewrite_suggestion)[:300]

        eid = str(uuid.uuid4())
        with self._lock:
            old = self._board.get(agent_id)
            if old is None:
                snap = AgentSnapshot(
                    agent_id=agent_id, modality=payload.get("modality", ""),
                    goal=payload.get("goal", ""),
                    aspect=payload.get("aspect"))
                self._board[agent_id] = snap
                delta = "first_report"
                version = 1
            else:
                snap = old
                delta = "superseded"
                version = old.version + 1
            snap.finding = finding
            snap.coverage_covered = cov_covered
            snap.coverage_gaps = cov_gaps
            snap.conf_retrieval = conf_retrieval
            snap.conf_agreement = conf_agreement
            snap.conf_coverage  = conf_coverage
            snap.ambiguity = ambiguity
            snap.n_retrieved = int(payload["n_retrieved"])
            snap.n_kept = n_kept
            snap.top_score = float(payload.get("top_score", 0.0))
            snap.score_spread = float(payload["score_spread"])
            snap.caveat = caveat
            snap.rationale = str(payload.get("rationale", ""))[:400]
            snap.suspicious_confidence = suspicious
            snap.missing_aspects = missing_aspects
            snap.low_support_claims = low_support
            snap.citations = cites
            snap.unverified_citations = unverified
            snap.evidence_mode = evidence_mode
            snap.retrieval_quality = retrieval_quality
            snap.modality_fit = modality_fit_val
            snap.modality_fit_reason = modality_fit_reason
            snap.query_rewrite_suggestion = query_rewrite_suggestion
            snap.version = version
            snap.last_updated = time.time()
            snap.delta = delta
            if snap.status == "running":
                snap.status = "done"
            self._trace.append({"id": eid, "agent_id": agent_id,
                                "stage": "summary", "payload": dict(payload),
                                "version": version, "ts": time.time()})
        return eid

    def _write_sketch(self, agent_id, payload):
        """Tier-2.5 — Evidence Sketch. Accepts two shapes:
          (legacy)  key_sentences: [{id, text, relevance}]
          (new)     key_candidates: [{id, relevance, evidence_hit,
                                      note?, text?}]
        `evidence_hit` is the sub-agent's estimate of how well this single
        candidate satisfies the subtask (0-1) — a per-candidate hit-rate
        signal the DA can use to decide whether INSPECT is worth it.
        Both shapes produce the same canonical stored form:
          key_candidates: [{id, relevance, evidence_hit, note, text}]
        where text/note may be empty strings for visual modality."""
        provided = _TIER25_ACCEPTED & set(payload.keys())
        if not provided:
            raise DisclosureError(
                f"sketch payload must contain one of: "
                f"{sorted(_TIER25_ACCEPTED)}")
        if len(provided) > 1:
            raise DisclosureError(
                f"sketch payload must contain exactly ONE of "
                f"{sorted(_TIER25_ACCEPTED)}, got both")
        shape = provided.pop()

        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._sketch_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for sketch; "
                        f"DA must issue REQUEST_EVIDENCE_SKETCH first")
                self._sketch_authorized.discard(agent_id)
            items = payload[shape]
            if not isinstance(items, list) or not items:
                raise DisclosureError(f"{shape} must be a non-empty list")
            known = self._known_ids.get(agent_id, set())
            cleaned = []
            for s in items[:6]:   # cap at 6
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id", ""))
                if not sid or sid not in known:
                    continue
                rel = float(s.get("relevance", 0.0))
                rel = max(0.0, min(1.0, rel))
                # evidence_hit is NEW (per-candidate hit-rate). Legacy
                # key_sentences payload doesn't supply it — default to
                # relevance as a reasonable proxy.
                hit_raw = s.get("evidence_hit", rel if shape == "key_sentences"
                                                else 0.0)
                try:
                    hit = max(0.0, min(1.0, float(hit_raw)))
                except (TypeError, ValueError):
                    hit = 0.0
                text = str(s.get("text", ""))[:300]
                note = str(s.get("note", ""))[:300]
                # For legacy sentences, require non-empty text. For new
                # key_candidates, text may be empty (visual modality).
                if shape == "key_sentences" and not text:
                    continue
                cleaned.append({"id": sid, "relevance": rel,
                                "evidence_hit": hit,
                                "text": text, "note": note})
            if not cleaned:
                raise DisclosureError(
                    "sketch has no usable entries (empty or all ids fabricated)")
            eid = str(uuid.uuid4())
            self._sketches[agent_id] = {"id": eid,
                                        "key_candidates": cleaned,
                                        "ts": time.time()}
            snap = self._board.get(agent_id)
            if snap:
                snap.sketch = {"key_candidates": cleaned}
            self._trace.append({"id": eid, "agent_id": agent_id,
                                "stage": "sketch",
                                "n_candidates": len(cleaned),
                                "ts": time.time()})
        return eid

    def get_sketch(self, agent_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._sketches.get(agent_id)

    def _write_curated(self, agent_id, payload, aspect_agreements=None):
        """Tier-2 curated disclosure with cross-modal evidence alignment.

        This is the CORE of progressive disclosure for EMNLP paper:
        - Cross-modal evidence filtering and unification
        - Highlights conflicting evidence across modalities
        - Deep dive for important-but-low-confidence subtasks

        Payload schema:
          {
            "key_candidates": [
              {"id": str, "relevance": float, "evidence_hit": float,
               "note": str?, "text": str?, "raw": str?}
            ],
            "with_raw": bool
          }

        Cross-Modal Analysis:
          - If this agent's aspect has cross-modal conflict, highlight it
          - Mark candidates that support conflicting positions
          - Provide structured comparison for DA's decision making

        Authorization:
          - agent must be in _curated_light_authorized
          - raw blobs only persisted when in _curated_raw_authorized
        """
        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._curated_light_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for curated; "
                        f"DA must issue REQUEST_CURATED_EVIDENCE first")
                self._curated_light_authorized.discard(agent_id)
                raw_auth = self._curated_raw_authorized.pop(agent_id, None)
            else:
                raw_auth = True

            cands_in = payload.get("key_candidates")
            if not isinstance(cands_in, list) or not cands_in:
                raise DisclosureError(
                    "curated.key_candidates must be a non-empty list")

            known = self._known_ids.get(agent_id, set())
            known_meta = dict(self._known_meta.get(agent_id, {}))

            # === Get agent's aspect and conflict status ===
            snap = self._board.get(agent_id)
            agent_aspect = snap.aspect if snap else None
            is_conflict_aspect = False
            conflict_details = None

            if aspect_agreements and agent_aspect:
                for agr in aspect_agreements:
                    if agr.get("aspect") == agent_aspect:
                        if agr.get("agreement_state") == "disagree":
                            is_conflict_aspect = True
                            conflict_details = agr.get("conflict_details", [])
                        break

            # === Cross-modal evidence processing ===
            def _raw_allowed(cid: str) -> bool:
                if raw_auth is True: return True
                if isinstance(raw_auth, set): return cid in raw_auth
                return False

            cleaned = []
            for c in cands_in[:10]:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("id", ""))
                if not cid or cid not in known:
                    continue

                rel = max(0.0, min(1.0, float(c.get("relevance", 0.0))))
                try:
                    hit = max(0.0, min(1.0, float(c.get("evidence_hit", rel))))
                except (TypeError, ValueError):
                    hit = rel

                text = str(c.get("text", ""))[:400]
                note = str(c.get("note", ""))[:300]
                raw_blob = c.get("raw")

                if raw_blob is not None and not _raw_allowed(cid):
                    raw_blob = None
                if raw_blob is not None:
                    raw_blob = str(raw_blob)[:60_000]

                entry = {
                    "id": cid,
                    "relevance": rel,
                    "evidence_hit": hit,
                    "text": text,
                    "note": note,
                    "meta": known_meta.get(cid, {})
                }

                if raw_blob is not None:
                    entry["raw"] = raw_blob

                # === Conflict Highlighting ===
                # If this is a conflict aspect, mark candidates that need DA attention
                if is_conflict_aspect:
                    # High-relevance candidates are flagged for cross-modal comparison
                    entry["_conflict_flag"] = rel > 0.7
                    entry["_supporting_position"] = note if note else text[:100]

                cleaned.append(entry)

            if not cleaned:
                raise DisclosureError(
                    "curated has no usable entries (empty or all fabricated)")

            with_raw_actual = any("raw" in e for e in cleaned)

            # === Cross-Modal Metadata ===
            cross_modal_meta = {
                "agent_aspect": agent_aspect,
                "is_conflict_aspect": is_conflict_aspect,
                "n_conflict_candidates": sum(1 for e in cleaned if e.get("_conflict_flag")),
            }

            if is_conflict_aspect and conflict_details:
                cross_modal_meta["conflict_summary"] = {
                    "n_modalities": len(conflict_details),
                    "modalities": list(set(d.get("modality") for d in conflict_details)),
                }

            eid = str(uuid.uuid4())
            self._curated[agent_id] = {
                "id": eid,
                "key_candidates": cleaned,
                "with_raw": with_raw_actual,
                "cross_modal_meta": cross_modal_meta,
                "ts": time.time(),
            }

            if snap:
                snap.curated = {
                    "n_candidates": len(cleaned),
                    "with_raw": with_raw_actual,
                    "is_conflict_aspect": is_conflict_aspect,
                }

            self._trace.append({
                "id": eid,
                "agent_id": agent_id,
                "stage": "curated",
                "n_candidates": len(cleaned),
                "with_raw": with_raw_actual,
                "cross_modal_meta": cross_modal_meta,
                "ts": time.time()
            })

        return eid

    def _write_full(self, agent_id, payload):
        missing = _TIER3_REQUIRED - set(payload.keys())
        if missing:
            raise DisclosureError(
                f"full missing fields: {sorted(missing)}")
        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._full_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for tier-3; "
                        f"Decision Agent must issue REQUEST_FULL_EVIDENCE first")
                self._full_authorized.discard(agent_id)
            eid = str(uuid.uuid4())
            self._archive.setdefault(agent_id, []).append({
                "id": eid, "content": payload["content"],
                "sources": list(payload["sources"]),
                "ts": time.time(),
            })
            snap = self._board.get(agent_id)
            if snap: snap.full_available = True
            self._trace.append({"id": eid, "agent_id": agent_id,
                                "stage": "full", "ts": time.time()})
        return eid

    # ---------------- read paths (DA + final answer) --------------------
    def status_board(self, include_terminated: bool = False) -> List[dict]:
        """The DA's default view. One row per active agent."""
        with self._lock:
            out = []
            for snap in self._board.values():
                if not include_terminated and snap.status in ("killed", "failed"):
                    continue
                out.append({
                    "agent_id": snap.agent_id,
                    "modality": snap.modality,
                    "goal": snap.goal,
                    "aspect": snap.aspect,
                    "status": snap.status,
                    "finding": snap.finding,
                    "coverage": {
                        "covered": round(snap.coverage_covered, 3),
                        "gaps": list(snap.coverage_gaps),
                    },
                    "confidence": {
                        "retrieval_score": (round(snap.conf_retrieval, 3)
                                           if snap.conf_retrieval is not None else None),
                        "evidence_agreement": (round(snap.conf_agreement, 3)
                                              if snap.conf_agreement is not None else None),
                        "coverage": (round(snap.conf_coverage, 3)
                                    if snap.conf_coverage is not None else None),
                    },
                    "ambiguity": (round(snap.ambiguity, 3)
                                  if snap.ambiguity is not None else None),
                    "n_retrieved": snap.n_retrieved,
                    "n_kept": snap.n_kept,
                    "top_score": round(snap.top_score, 3),
                    "score_spread": round(snap.score_spread, 3),
                    "caveat": snap.caveat,
                    "suspicious": snap.suspicious_confidence,
                    "missing_aspects": list(snap.missing_aspects),
                    "low_support_claims": list(snap.low_support_claims),
                    "citations": list(snap.citations),
                    "unverified_citations": list(snap.unverified_citations),
                    "evidence_mode": snap.evidence_mode,
                    "retrieval_quality": snap.retrieval_quality,
                    "modality_fit": snap.modality_fit,
                    "modality_fit_reason": snap.modality_fit_reason,
                    "query_rewrite_suggestion": snap.query_rewrite_suggestion,
                    "sketch_available": snap.sketch is not None,
                    "curated_available": snap.curated is not None,
                    "curated_summary": dict(snap.curated) if snap.curated else None,
                    "full_available": snap.full_available,
                    "delta": snap.delta,
                    "version": snap.version,
                })
            return out

    def consume_deltas(self) -> None:
        """Called by Orchestrator AFTER the DA reads the board, so next turn
        shows 'unchanged' instead of stale first_report/superseded."""
        with self._lock:
            for snap in self._board.values():
                snap.delta = "unchanged"

    def get_archive(self, agent_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._archive.get(agent_id, []))

    def archived_agents(self) -> List[str]:
        with self._lock:
            return list(self._archive.keys())

    # ---------------- metrics ------------------------------------------
    def coverage(self) -> float:
        """Fraction of non-terminated agents that have reached tier-2."""
        with self._lock:
            alive = [s for s in self._board.values()
                     if s.status not in ("killed", "failed")]
            if not alive: return 0.0
            with_summary = [s for s in alive if s.version > 0]
            return len(with_summary) / len(alive)

    def max_confidence(self) -> float:
        """Return the max of any confidence axis across live agents.
        Used as the final answer's reported confidence and for DA stopping rules."""
        with self._lock:
            alive = [s for s in self._board.values()
                     if s.status not in ("killed", "failed")]
            best = 0.0
            for s in alive:
                for v in (s.conf_retrieval, s.conf_agreement, s.conf_coverage):
                    if v is not None and v > best:
                        best = v
            return best

    def aspect_agreements(self) -> List[Dict[str, Any]]:
        """P2 — group agents by aspect and compute cross-modal agreement.

        Returns one row per aspect that has ≥1 sub-agent reporting summary.
        agreement_state semantics:
          single_source — only one agent on this aspect
          agree         — multi-agent, all confidences ≥ 0.6, max gap < 0.3,
                          findings token-overlap > 0.3
          disagree      — multi-agent, all confidences ≥ 0.6, AND
                          (max gap > 0.4 OR token-overlap < 0.15)
          complementary — multi-agent, neither agree nor disagree

        Notes:
          - finding similarity is naive Jaccard — a WEAK signal, used only
            with confidence as a corroborator. Never alone.
          - any_exhausted aggregates retrieval_quality across the group so
            DA can avoid wasted CONTINUE_RETRIEVAL on saturated indices.
        """
        def _max_conf(s):
            vals = [s.conf_retrieval, s.conf_agreement, s.conf_coverage]
            return max((v for v in vals if v is not None), default=0.0)
        def _tokens(s):
            import re
            return set(re.findall(r"[A-Za-z0-9]+", (s or "").lower()))
        def _jaccard(a, b):
            ta, tb = _tokens(a), _tokens(b)
            if not ta or not tb: return 0.0
            return len(ta & tb) / max(len(ta | tb), 1)

        with self._lock:
            by_aspect: Dict[str, List[AgentSnapshot]] = {}
            for s in self._board.values():
                if s.status in ("killed", "failed") or s.version == 0:
                    continue
                key = s.aspect or f"_modality_{s.modality}"  # group fallback
                by_aspect.setdefault(key, []).append(s)

            out: List[Dict[str, Any]] = []
            for aspect, agents in by_aspect.items():
                confs = [_max_conf(a) for a in agents]
                covs = [a.coverage_covered for a in agents]
                mods = sorted({a.modality for a in agents})
                exhausted = any(a.retrieval_quality == "exhausted"
                                for a in agents)
                misfit = any(a.modality_fit is False for a in agents)

                if len(agents) <= 1:
                    state = "single_source"
                    max_gap = 0.0
                else:
                    max_gap = max(confs) - min(confs)
                    # Pairwise jaccard average over findings.
                    pair_sims = []
                    for i in range(len(agents)):
                        for j in range(i+1, len(agents)):
                            pair_sims.append(
                                _jaccard(agents[i].finding, agents[j].finding))
                    avg_sim = sum(pair_sims) / max(len(pair_sims), 1)
                    all_confident = all(c >= 0.6 for c in confs)
                    if all_confident and max_gap < 0.3 and avg_sim > 0.3:
                        state = "agree"
                    elif all_confident and (max_gap > 0.4 or avg_sim < 0.15):
                        state = "disagree"
                    else:
                        state = "complementary"

                # Round 4 — when agents disagree, capture each side's
                # finding so DA can RESOLVE_CONFLICT without having to
                # INSPECT raw evidence first.
                conflict_details = None
                if state == "disagree":
                    conflict_details = [
                        {"agent_id": a.agent_id,
                         "modality": a.modality,
                         "finding": (a.finding or "")[:200],
                         "max_conf": round(_max_conf(a), 3)}
                        for a in agents]

                out.append({
                    "aspect": aspect,
                    "n_agents": len(agents),
                    "agent_ids": [a.agent_id for a in agents],
                    "modalities_covered": mods,
                    "agreement_state": state,
                    "max_conf_gap": round(max_gap, 3),
                    "median_coverage": round(
                        sorted(covs)[len(covs)//2] if covs else 0.0, 3),
                    "any_exhausted": exhausted,
                    "any_modality_misfit": misfit,
                    "conflict_details": conflict_details,
                })
            return out

    # ---------------- trace (for debugging / ablation studies) ----------
    def trace(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._trace)
