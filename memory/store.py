"""Multi-index retrieval store with hybrid dense + sparse fusion.

Each modality has:
  - a dense index (vector similarity, offline-embedded)
  - an optional sparse index (BM25 over content + caption/OCR text)

When `hybrid=True`, `search()` fuses both with Reciprocal Rank Fusion
(Cormack et al. 2009):

    RRF(d) = sum over rankers r of  1 / (K + rank_r(d))

RRF is rank-based, so it's robust to scale differences between dense
similarity (0-1) and BM25 scores (unbounded).

When `hybrid=False` (default), falls back to the dense index only, keeping
the tests and the offline BoW fallback working.
"""
from typing import List, Dict, Any, Callable, Optional
import math, re
from collections import Counter

MODALITIES = ("doc_text", "doc_visual", "video_text", "video_visual")


def _tokens(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", (s or "").lower())


# ---------------------------------------------------------------- BM25 index
class BM25Index:
    """In-memory BM25 over document content + meta values.
    Works without external dependencies so tests remain self-contained."""

    def __init__(self, docs: List[Dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1 = k1
        self.b = b
        self._doc_tokens: List[List[str]] = []
        for d in docs:
            hay = (d.get("content") or "") + " " + \
                  " ".join(str(v) for v in (d.get("meta") or {}).values())
            self._doc_tokens.append(_tokens(hay))
        self._doc_len = [len(t) for t in self._doc_tokens]
        self._avgdl = (sum(self._doc_len) / len(self._doc_len)
                       if self._doc_len else 0)
        # Document frequency
        self._df: Counter = Counter()
        for toks in self._doc_tokens:
            for t in set(toks):
                self._df[t] += 1
        self._N = len(docs)

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        if n == 0: return 0.0
        # BM25 plus smoothing
        return math.log(1 + (self._N - n + 0.5) / (n + 0.5))

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = _tokens(query)
        if not q or not self.docs: return []
        scores = [0.0] * len(self.docs)
        for i, toks in enumerate(self._doc_tokens):
            if not toks: continue
            tf = Counter(toks)
            dl = self._doc_len[i]
            s = 0.0
            for term in q:
                if term not in tf: continue
                idf = self._idf(term)
                freq = tf[term]
                norm = 1 - self.b + self.b * dl / max(self._avgdl, 1)
                s += idf * (freq * (self.k1 + 1)) / (freq + self.k1 * norm)
            scores[i] = s
        ranked = [(s, d) for s, d in zip(scores, self.docs) if s > 0]
        ranked.sort(key=lambda x: x[0], reverse=True)
        return [{**d, "score": s} for s, d in ranked[:k]]


# ---------------------------------------------------------------- Dense index
class VectorIndex:
    """Single-modality dense index. Default BoW score for offline testing.
    In production, inject `encoder` and `faiss_index`."""

    def __init__(self, modality: str, docs: List[Dict[str, Any]],
                 encoder: Optional[Callable[[str], List[float]]] = None):
        assert modality in MODALITIES, f"unknown modality: {modality}"
        self.modality = modality
        self.docs = docs
        self.encoder = encoder

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q = _tokens(query)
        scored = []
        for d in self.docs:
            hay = (d.get("content") or "") + " " + \
                  " ".join(str(v) for v in (d.get("meta") or {}).values())
            t = _tokens(hay)
            if not q or not t:
                s = 0.0
            else:
                overlap = sum(1 for x in q if x in t)
                s = overlap / max(len(q), 1)
            if s > 0:
                scored.append({**d, "modality": self.modality, "score": s})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]


# ---------------------------------------------------------------- RRF
def rrf_merge(rankings: List[List[Dict[str, Any]]], k: int = 5,
              K: int = 60) -> List[Dict[str, Any]]:
    """Reciprocal Rank Fusion. K=60 is the value from Cormack et al. 2009."""
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}
    for lst in rankings:
        for rank, item in enumerate(lst):
            did = item.get("id") or str(id(item))
            scores[did] = scores.get(did, 0.0) + 1.0 / (K + rank)
            # Keep the richest representation seen.
            if did not in docs or len(str(item)) > len(str(docs[did])):
                docs[did] = item
    out = []
    for did, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]:
        out.append({**docs[did], "rrf_score": s})
    return out


# ---------------------------------------------------------------- Store
class MultiIndexStore:
    """Routes queries to the correct (modality, channel) index.
    If hybrid=True, merges dense + BM25 with RRF per modality."""

    def __init__(self, indices: Dict[str, VectorIndex],
                 hybrid: bool = True):
        unknown = set(indices) - set(MODALITIES)
        if unknown:
            raise ValueError(f"unknown modalities: {unknown}")
        self._indices = indices
        self._hybrid = hybrid
        # Build sparse companions for each modality when hybrid is on.
        self._sparse: Dict[str, BM25Index] = {}
        if hybrid:
            for mod, idx in indices.items():
                self._sparse[mod] = BM25Index(idx.docs)

    def search(self, query: str, modality: str, k: int = 5,
               **kw) -> List[Dict[str, Any]]:
        if modality not in self._indices:
            raise KeyError(f"no index for modality={modality}; "
                           f"available: {list(self._indices)}")
        dense_hits = self._indices[modality].search(query=query, k=k * 2)
        if not self._hybrid:
            return dense_hits[:k]
        sparse_hits = self._sparse[modality].search(query=query, k=k * 2)
        fused = rrf_merge([dense_hits, sparse_hits], k=k)
        # Re-attach modality field (rrf_merge preserves dense's if there).
        return [{**h, "modality": modality} for h in fused]

    def available_modalities(self) -> List[str]:
        return list(self._indices.keys())
