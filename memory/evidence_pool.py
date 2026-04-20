"""Evidence pool — progressive disclosure model tuned for Decision-Agent attention.

LAYER A  Scratch        — sub-agent private working area (candidates, rerank
                          intermediates). DA never sees it.
LAYER B  Status Board   — one AgentSnapshot per sub-agent, SUPERSEDE on write.
                          This is the DA's default view (Tier-1).
LAYER C  Archive        — raw Tier-3 evidence keyed by agent_id. DA queries
                          this on demand during final answer synthesis.

DISCLOSURE TIERS (progressive):
  Tier-1 (summary)   — Default view. Structured finding + task completion + local gaps.
  Tier-2 (sketch)    — 2-5 key evidence excerpts on DA's REQUEST_EVIDENCE_SKETCH.
  Tier-3 (full)      — Complete raw content on DA's REQUEST_FULL_EVIDENCE.

Intent entries are retained in the Orchestrator's trace for debugging but are
NOT part of progressive disclosure — they add noise without improving DA decisions.
"""
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Set
import threading, time, uuid

# -- Tier schemas --
# Tier-1 = summary. Default view for the Decision Agent. Structured so the DA
# can quickly understand "what was found / what's missing / how confident" without
# seeing raw evidence.
_TIER1_REQUIRED = {
    "finding",           # Direct answer to the subtask
    "reasoning",         # Why these evidence support the conclusion
    "task_completion",   # {addressed, partial, uncovered}
    "confidence",        # {retrieval_quality, evidence_coherence, reasoning_strength}
    "local_gaps",        # {critical, suggested_modalities}
    "citations",         # Verified evidence ids
    "n_retrieved",       # Total candidates retrieved
    "n_kept",            # After reranking
}
# Tier-2 = evidence sketch. Returned ONLY when DA issues REQUEST_EVIDENCE_SKETCH.
# Sits between Tier-1 (summary) and Tier-3 (full). DA can inspect 2-5 key excerpts
# without paying the full cost.
# Accepts either shape:
#   {"key_sentences": [{"id","text","relevance"}]}    — legacy, text-only
#   {"key_candidates": [{"id","relevance","evidence_hit",
#                        "note"?, "text"?}]}          — new, works for text + visual
# Exactly ONE of the two keys must be present.
_TIER2_ACCEPTED = {"key_sentences", "key_candidates"}
# Tier-3 = full content. Only after REQUEST_FULL_EVIDENCE authorization.
_TIER3_REQUIRED = {"content", "sources"}

# Internal intent stage (not shown to DA)
_INTENT_REQUIRED = {"modality", "data_source", "planned_k"}

# Optional reflection signals
EVIDENCE_MODE_ENUM = {None, "raw_visual", "caption_only", "text_native"}
RETRIEVAL_QUALITY_ENUM = {None, "exhausted", "partial", "thin"}

# Sub-agent uses coarse enums for confidence axes; 7B models estimate
# discrete buckets more reliably than decimals. Orchestrator maps to floats.
_CONF_AXIS_ENUM = {"high", "medium", "low", "unclear", None}
_CONF_ENUM_TO_FLOAT = {"high": 0.85, "medium": 0.55, "low": 0.25,
                      "unclear": None, None: None}


class DisclosureError(Exception):
    """Raised on tier schema violation or unauthorized Tier-3 write."""


@dataclass
class AgentSnapshot:
    """One row of the DA's Status Board — the current latest report
    from a sub-agent. Overwritten on each new Tier-1 write.

    Tier-1 summary focuses on:
      - finding: Direct answer to the subtask
      - reasoning: Why the evidence supports this conclusion
      - task_completion: Clear breakdown of addressed/partial/uncovered
      - confidence: Three-axis (retrieval_quality, evidence_coherence, reasoning_strength)
      - local_gaps: What THIS modality cannot answer (DA decision key)
      - sketch: Tier-2 key excerpts (populated on authorized request)
      - full_available: Whether Tier-3 has been written
    """
    agent_id: str
    modality: str
    goal: str
    aspect: Optional[str] = None               # DA-assigned aspect tag
    status: str = "running"

    # -- Tier-1: Summary --
    finding: str = ""
    reasoning: str = ""
    # task_completion: three-category breakdown
    tc_addressed: List[str] = field(default_factory=list)
    tc_partial: List[str] = field(default_factory=list)
    tc_uncovered: List[str] = field(default_factory=list)
    # confidence: three-axis redefined for clarity
    conf_retrieval_quality: Optional[float] = None
    conf_evidence_coherence: Optional[float] = None
    conf_reasoning_strength: Optional[float] = None
    # local_gaps: what this modality cannot answer
    local_gaps_critical: List[str] = field(default_factory=list)
    local_gaps_suggested: List[str] = field(default_factory=list)
    # retrieval metadata (kept for diagnostics)
    n_retrieved: int = 0
    n_kept: int = 0
    top_score: float = 0.0
    score_spread: float = 0.0
    # citations
    citations: List[Dict[str, Any]] = field(default_factory=list)
    unverified_citations: List[str] = field(default_factory=list)

    # -- Reflection signals (optional, helps DA pick repair actions) --
    evidence_mode: Optional[str] = None
        # "raw_visual"  — VLM truly saw the image/frame
        # "caption_only" — only had the caption text
        # "text_native"  — text modality, content IS the evidence
    retrieval_quality: Optional[str] = None
        # "exhausted" — this query+index is wrung out; CONTINUE is wasted
        # "partial"   — more available, needs query rewrite
        # "thin"      — this modality wrong for this goal; SWITCH advised
    modality_fit: Optional[bool] = None
    modality_fit_reason: str = ""
    query_rewrite_suggestion: Optional[str] = None

    # -- Progressive disclosure slots --
    sketch: Optional[Dict[str, Any]] = None    # Tier-2 key_sentences
    sketch_authorized: bool = False            # has DA issued REQUEST_EVIDENCE_SKETCH?
    curated: Optional[Dict[str, Any]] = None   # Tier-2 curated (cross-modal)
    full_available: bool = False               # has Tier-3 been written?

    # -- Versioning --
    version: int = 0
    last_updated: float = field(default_factory=time.time)
    delta: str = "first_report"

    # -- Legacy compatibility (deprecated, will be removed) --
    caveat: Optional[str] = None
    rationale: str = ""
    suspicious_confidence: bool = False
    missing_aspects: List[str] = field(default_factory=list)
    low_support_claims: List[str] = field(default_factory=list)
    ambiguity: Optional[float] = None


class EvidencePool:
    """Progressive disclosure store. Thread-safe. Supersede semantics on the board."""

    def __init__(self, bypass_tier3_gate: bool = False,
                 finding_max_chars: int = 300):
        self._lock = threading.Lock()
        self._board: Dict[str, AgentSnapshot] = {}
        self._archive: Dict[str, List[Dict[str, Any]]] = {}
        self._sketches: Dict[str, Dict[str, Any]] = {}   # agent_id → sketch
        self._trace: List[Dict[str, Any]] = []
        self._full_authorized: Set[str] = set()
        self._sketch_authorized: Set[str] = set()
        # Curated authorizations (cross-modal Tier-2)
        self._curated_light_authorized: Set[str] = set()
        self._curated_raw_authorized: Dict[str, Any] = {}
        self._curated: Dict[str, Dict[str, Any]] = {}   # agent_id → payload
        self._known_ids: Dict[str, Set[str]] = {}
        # Per-agent map id → meta dict, populated by note_retrieved_candidates()
        self._known_meta: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._bypass_tier3_gate = bypass_tier3_gate
        self._finding_max_chars = finding_max_chars
        if bypass_tier3_gate:
            import warnings
            warnings.warn(
                "EvidencePool Tier-3 gate DISABLED (ablation mode).",
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
        with their asset_type/source/page/t."""
        with self._lock:
            known_ids = self._known_ids.setdefault(agent_id, set())
            known_meta = self._known_meta.setdefault(agent_id, {})
            for c in candidates:
                cid = str(c.get("id", ""))
                if not cid:
                    continue
                known_ids.add(cid)
                meta = c.get("meta") or {}
                known_meta[cid] = {
                    "asset_type": meta.get("asset_type"),
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
        """Grant one-time Tier-2 write permission."""
        with self._lock:
            self._sketch_authorized.add(agent_id)
            snap = self._board.get(agent_id)
            if snap: snap.sketch_authorized = True

    def is_authorized_for_sketch(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._sketch_authorized

    def authorize_full(self, agent_id: str) -> None:
        """Grant one-time Tier-3 write permission."""
        with self._lock:
            self._full_authorized.add(agent_id)

    def is_authorized_for_full(self, agent_id: str) -> bool:
        with self._lock:
            return agent_id in self._full_authorized

    def authorize_curated(self, agent_id: str,
                          with_raw: bool = False,
                          ids: Optional[List[str]] = None) -> None:
        """DA says: this sub-agent may write a `curated` block (Tier-2)."""
        with self._lock:
            self._curated_light_authorized.add(agent_id)
            if with_raw:
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
        """Internal intent stage (stored in trace, NOT shown to DA)."""
        missing = _INTENT_REQUIRED - set(payload.keys())
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
        """Tier-1 summary — DA's default view."""
        # Required fields check
        missing = _TIER1_REQUIRED - set(payload.keys())
        if missing:
            raise DisclosureError(
                f"Tier-1 summary missing fields: {sorted(missing)}")

        # finding: direct answer to subtask
        finding = str(payload["finding"])[:self._finding_max_chars]

        # reasoning: why these evidence support the conclusion
        reasoning = str(payload.get("reasoning", ""))[:600]

        # task_completion: three-category breakdown
        tc = payload.get("task_completion", {})
        if not isinstance(tc, dict):
            raise DisclosureError("task_completion must be an object")
        tc_addressed = [str(x)[:200] for x in tc.get("addressed", [])[:6]]
        tc_partial = [str(x)[:200] for x in tc.get("partial", [])[:6]]
        tc_uncovered = [str(x)[:200] for x in tc.get("uncovered", [])[:6]]

        # confidence: three-axis (redefined)
        conf = payload.get("confidence", {})
        if not isinstance(conf, dict):
            raise DisclosureError("confidence must be an object")
        def _axis(name, required=True):
            v = conf.get(name)
            if v is None:
                if required:
                    raise DisclosureError(f"confidence.{name} is required")
                return None
            if isinstance(v, (int, float)):
                f = float(v)
                if not 0.0 <= f <= 1.0:
                    raise DisclosureError(f"confidence.{name} must be in [0,1]")
                return f
            if isinstance(v, str):
                if v not in _CONF_AXIS_ENUM:
                    raise DisclosureError(
                        f"confidence.{name} enum must be one of "
                        f"{sorted(x for x in _CONF_AXIS_ENUM if x)}")
                return _CONF_ENUM_TO_FLOAT[v]
            raise DisclosureError(f"confidence.{name} must be number or enum string")
        conf_rq = _axis("retrieval_quality")
        conf_ec = _axis("evidence_coherence")
        conf_rs = _axis("reasoning_strength")

        # local_gaps: what this modality cannot answer
        gaps = payload.get("local_gaps", {})
        if not isinstance(gaps, dict):
            raise DisclosureError("local_gaps must be an object")
        gaps_critical = [str(x)[:200] for x in gaps.get("critical", [])[:6]]
        gaps_suggested = [str(x)[:30] for x in gaps.get("suggested_modalities", [])[:4]]

        # citations
        raw_cites = payload.get("citations", [])
        if not isinstance(raw_cites, list):
            raise DisclosureError("citations must be a list")
        n_kept = int(payload.get("n_kept", 0))
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
            merged = dict(known_meta.get(cid, {}))
            merged.update({k: v for k, v in user_meta.items() if v is not None})
            merged["id"] = cid
            cites.append(merged)

        if n_kept > 0 and not cites:
            raise DisclosureError(
                f"Tier-1 from {agent_id} has no verifiable citations; "
                f"provided={[str(c)[:40] for c in raw_cites]}, "
                f"known retrieval ids={sorted(known)[:10]}")

        # Reflection signals (optional)
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

        # Legacy support (deprecated)
        n_retrieved = int(payload.get("n_retrieved", n_kept))
        top_score = float(payload.get("top_score", 0.0))
        score_spread = float(payload.get("score_spread", 0.0))
        caveat = payload.get("caveat")

        eid = str(uuid.uuid4())
        with self._lock:
            old = self._board.get(agent_id)
            if old is None:
                snap = AgentSnapshot(
                    agent_id=agent_id,
                    modality=payload.get("modality", ""),
                    goal=payload.get("goal", ""),
                    aspect=payload.get("aspect"))
                self._board[agent_id] = snap
                delta = "first_report"
                version = 1
            else:
                snap = old
                delta = "superseded"
                version = old.version + 1

            # Update Tier-1 fields
            snap.finding = finding
            snap.reasoning = reasoning
            snap.tc_addressed = tc_addressed
            snap.tc_partial = tc_partial
            snap.tc_uncovered = tc_uncovered
            snap.conf_retrieval_quality = conf_rq
            snap.conf_evidence_coherence = conf_ec
            snap.conf_reasoning_strength = conf_rs
            snap.local_gaps_critical = gaps_critical
            snap.local_gaps_suggested = gaps_suggested
            snap.n_retrieved = n_retrieved
            snap.n_kept = n_kept
            snap.top_score = top_score
            snap.score_spread = score_spread
            snap.citations = cites
            snap.unverified_citations = unverified
            snap.evidence_mode = evidence_mode
            snap.retrieval_quality = retrieval_quality
            snap.modality_fit = modality_fit_val
            snap.modality_fit_reason = modality_fit_reason
            snap.query_rewrite_suggestion = query_rewrite_suggestion

            # Legacy fields (deprecated)
            snap.caveat = caveat
            snap.missing_aspects = tc_uncovered  # Map for backward compat
            snap.ambiguity = (min(1.0, score_spread * 2.0)
                             if n_kept >= 2 else 0.5)

            snap.version = version
            snap.last_updated = time.time()
            snap.delta = delta
            if snap.status == "running":
                snap.status = "done"

            self._trace.append({
                "id": eid, "agent_id": agent_id,
                "stage": "summary", "payload": dict(payload),
                "version": version, "ts": time.time()
            })
        return eid

    def _write_sketch(self, agent_id, payload):
        """Tier-2 — Evidence Sketch. Accepts two shapes:
          (legacy)  key_sentences: [{id, text, relevance}]
          (new)     key_candidates: [{id, relevance, evidence_hit, note?, text?}]
        `evidence_hit` is per-candidate hit-rate (0-1) signal the DA can use to
        decide whether deeper inspection is worth it.
        """
        provided = _TIER2_ACCEPTED & set(payload.keys())
        if not provided:
            raise DisclosureError(
                f"Tier-2 sketch payload must contain one of: "
                f"{sorted(_TIER2_ACCEPTED)}")
        if len(provided) > 1:
            raise DisclosureError(
                f"Tier-2 sketch payload must contain exactly ONE of "
                f"{sorted(_TIER2_ACCEPTED)}, got both")
        shape = provided.pop()

        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._sketch_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for Tier-2 sketch; "
                        f"DA must issue REQUEST_EVIDENCE_SKETCH first")
                self._sketch_authorized.discard(agent_id)
            items = payload[shape]
            if not isinstance(items, list) or not items:
                raise DisclosureError(f"{shape} must be a non-empty list")
            known = self._known_ids.get(agent_id, set())
            cleaned = []
            for s in items[:6]:
                if not isinstance(s, dict):
                    continue
                sid = str(s.get("id", ""))
                if not sid or sid not in known:
                    continue
                rel = float(s.get("relevance", 0.0))
                rel = max(0.0, min(1.0, rel))
                hit_raw = s.get("evidence_hit", rel if shape == "key_sentences" else 0.0)
                try:
                    hit = max(0.0, min(1.0, float(hit_raw)))
                except (TypeError, ValueError):
                    hit = 0.0
                text = str(s.get("text", ""))[:300]
                note = str(s.get("note", ""))[:300]
                if shape == "key_sentences" and not text:
                    continue
                cleaned.append({"id": sid, "relevance": rel,
                                "evidence_hit": hit,
                                "text": text, "note": note})
            if not cleaned:
                raise DisclosureError(
                    "Tier-2 sketch has no usable entries (empty or all ids fabricated)")
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
        """Tier-2 curated disclosure with cross-modal evidence alignment."""
        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._curated_light_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for Tier-2 curated; "
                        f"DA must issue REQUEST_CURATED_EVIDENCE first")
                self._curated_light_authorized.discard(agent_id)
                raw_auth = self._curated_raw_authorized.pop(agent_id, None)
            else:
                raw_auth = True

            cands_in = payload.get("key_candidates")
            if not isinstance(cands_in, list) or not cands_in:
                raise DisclosureError(
                    "Tier-2 curated.key_candidates must be a non-empty list")

            known = self._known_ids.get(agent_id, set())
            known_meta = dict(self._known_meta.get(agent_id, {}))

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
                    "id": cid, "relevance": rel, "evidence_hit": hit,
                    "text": text, "note": note, "meta": known_meta.get(cid, {})
                }
                if raw_blob is not None:
                    entry["raw"] = raw_blob
                if is_conflict_aspect:
                    entry["_conflict_flag"] = rel > 0.7
                    entry["_supporting_position"] = note if note else text[:100]
                cleaned.append(entry)

            if not cleaned:
                raise DisclosureError(
                    "Tier-2 curated has no usable entries (empty or all fabricated)")

            with_raw_actual = any("raw" in e for e in cleaned)
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
                "id": eid, "agent_id": agent_id,
                "stage": "curated",
                "n_candidates": len(cleaned),
                "with_raw": with_raw_actual,
                "cross_modal_meta": cross_modal_meta,
                "ts": time.time()
            })
        return eid

    def _write_full(self, agent_id, payload):
        """Tier-3 — Full content."""
        missing = _TIER3_REQUIRED - set(payload.keys())
        if missing:
            raise DisclosureError(
                f"Tier-3 full missing fields: {sorted(missing)}")
        with self._lock:
            if not self._bypass_tier3_gate:
                if agent_id not in self._full_authorized:
                    raise DisclosureError(
                        f"agent {agent_id} not authorized for Tier-3; "
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
        """The DA's default view (Tier-1). One row per active agent."""
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
                    # Tier-1: Summary
                    "finding": snap.finding,
                    "reasoning": snap.reasoning,
                    "task_completion": {
                        "addressed": list(snap.tc_addressed),
                        "partial": list(snap.tc_partial),
                        "uncovered": list(snap.tc_uncovered),
                    },
                    "confidence": {
                        "retrieval_quality": (round(snap.conf_retrieval_quality, 3)
                                             if snap.conf_retrieval_quality is not None else None),
                        "evidence_coherence": (round(snap.conf_evidence_coherence, 3)
                                              if snap.conf_evidence_coherence is not None else None),
                        "reasoning_strength": (round(snap.conf_reasoning_strength, 3)
                                              if snap.conf_reasoning_strength is not None else None),
                    },
                    "local_gaps": {
                        "critical": list(snap.local_gaps_critical),
                        "suggested_modalities": list(snap.local_gaps_suggested),
                    },
                    "n_retrieved": snap.n_retrieved,
                    "n_kept": snap.n_kept,
                    "top_score": round(snap.top_score, 3),
                    "score_spread": round(snap.score_spread, 3),
                    "citations": list(snap.citations),
                    "unverified_citations": list(snap.unverified_citations),
                    # Reflection signals
                    "evidence_mode": snap.evidence_mode,
                    "retrieval_quality": snap.retrieval_quality,
                    "modality_fit": snap.modality_fit,
                    "modality_fit_reason": snap.modality_fit_reason,
                    "query_rewrite_suggestion": snap.query_rewrite_suggestion,
                    # Progressive disclosure flags
                    "sketch_available": snap.sketch is not None,
                    "curated_available": snap.curated is not None,
                    "curated_summary": dict(snap.curated) if snap.curated else None,
                    "full_available": snap.full_available,
                    # Versioning
                    "delta": snap.delta,
                    "version": snap.version,
                })
            return out

    def consume_deltas(self) -> None:
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
        """Fraction of non-terminated agents that have Tier-1 summary."""
        with self._lock:
            alive = [s for s in self._board.values()
                     if s.status not in ("killed", "failed")]
            if not alive: return 0.0
            with_summary = [s for s in alive if s.version > 0]
            return len(with_summary) / len(alive)

    def max_confidence(self) -> float:
        """Return the max confidence across live agents."""
        with self._lock:
            alive = [s for s in self._board.values()
                     if s.status not in ("killed", "failed")]
            best = 0.0
            for s in alive:
                for v in (s.conf_retrieval_quality, s.conf_evidence_coherence,
                          s.conf_reasoning_strength):
                    if v is not None and v > best:
                        best = v
            return best

    def aspect_agreements(self) -> List[Dict[str, Any]]:
        """Group agents by aspect and compute cross-modal agreement."""
        def _max_conf(s):
            vals = [s.conf_retrieval_quality, s.conf_evidence_coherence,
                    s.conf_reasoning_strength]
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
                key = s.aspect or f"_modality_{s.modality}"
                by_aspect.setdefault(key, []).append(s)

            out: List[Dict[str, Any]] = []
            for aspect, agents in by_aspect.items():
                confs = [_max_conf(a) for a in agents]
                covs = [len(a.tc_addressed) / max(len(a.tc_addressed) +
                                                 len(a.tc_partial) +
                                                 len(a.tc_uncovered), 1)
                        if a else 0 for a in agents]
                mods = sorted({a.modality for a in agents})
                exhausted = any(a.retrieval_quality == "exhausted" for a in agents)
                misfit = any(a.modality_fit is False for a in agents)

                if len(agents) <= 1:
                    state = "single_source"
                    max_gap = 0.0
                else:
                    max_gap = max(confs) - min(confs)
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

                conflict_details = None
                if state == "disagree":
                    conflict_details = [
                        {"agent_id": a.agent_id, "modality": a.modality,
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
                    "median_coverage": round(sorted(covs)[len(covs)//2] if covs else 0.0, 3),
                    "any_exhausted": exhausted,
                    "any_modality_misfit": misfit,
                    "conflict_details": conflict_details,
                })
            return out

    def trace(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._trace)
