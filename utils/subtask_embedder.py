"""Subtask embedder — turns subtask descriptions into vectors.

Purpose A (implemented): dedupe — drop near-duplicate subtasks so DA's
  decomposition doesn't fan out into 5 versions of the same question.

Purpose B (interface only, consumer TBD in Round 3): query alignment —
  use a subtask's embedding to query the retrieval index. For visual
  subtasks this means embedding in CLIP's text space so vectors can be
  compared to pre-indexed image vectors directly.

Design choice:
  All subtasks (text + visual) are embedded with the SAME encoder — a
  CLIP-text encoder. This gives us one comparable space for cross-modal
  dedup (purpose A) and lets visual subtasks query CLIP image indices
  directly (purpose B). Text subtasks can't query BGE text indices with
  these vectors, but purpose B for text is not wired up yet.

Production substitution:
  `CLIPTextEmbedder` wraps an actual CLIP text model (e.g. ViT-B/32 or
  SigLIP). For tests and for this dev environment (no GPU, no CLIP
  weights) we ship `MockCLIPTextEmbedder` — a deterministic BoW-hash
  embedder that preserves enough similarity structure for unit tests.
  Swap by passing a different embedder to `embed_subtasks(...)`.
"""
from __future__ import annotations
from typing import List, Optional, Protocol, Sequence
import hashlib
import math
import re


# =========================================================================
# Interface
# =========================================================================
class Embedder(Protocol):
    """Minimal contract an embedder must satisfy."""
    dim: int
    def embed(self, text: str) -> List[float]: ...
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]: ...


# =========================================================================
# Mock embedder — deterministic, dependency-free, for tests & dev
# =========================================================================
class MockCLIPTextEmbedder:
    """Hash-based pseudo-embedder. Similar inputs → similar outputs via
    bag-of-words hashing, which is good enough for dedup tests and lets
    production code run without a GPU/CLIP model installed.

    NOT for real evaluation. Production deployments should inject a real
    CLIP text encoder via the `Embedder` protocol.
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
        vec = [0.0] * self.dim
        for tok in tokens:
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            for i in range(4):   # multi-hash so similar strings share buckets
                idx = (h >> (i * 8)) % self.dim
                vec[idx] += 1.0
        # L2 normalise so cosine = dot product.
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


# =========================================================================
# CLIP text embedder — real implementation stub
# =========================================================================
class CLIPTextEmbedder:
    """Production CLIP text encoder. Lazy-imports `transformers` / `torch`
    so the base system doesn't require them to be installed.

    Usage:
      emb = CLIPTextEmbedder(model_name="openai/clip-vit-base-patch32")
      vecs = emb.embed_batch([s.description for s in subtasks])
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: Optional[str] = None):
        try:
            import torch
            from transformers import CLIPTokenizer, CLIPTextModel
        except ImportError as e:
            raise RuntimeError(
                "CLIPTextEmbedder needs `transformers` and `torch`. "
                "Install with: pip install transformers torch") from e
        self._torch = torch
        self._tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self._model = CLIPTextModel.from_pretrained(model_name)
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device).eval()
        # Peek one config to learn the dim
        self.dim = self._model.config.hidden_size

    def embed(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        import torch
        with torch.no_grad():
            inputs = self._tokenizer(list(texts), padding=True,
                                     truncation=True, max_length=77,
                                     return_tensors="pt").to(self._device)
            out = self._model(**inputs)
            # Pooler output is the [EOS]-token representation — the standard
            # CLIP text embedding. Normalise for cosine = dot.
            emb = out.pooler_output
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return emb.cpu().tolist()


# =========================================================================
# Subtask operations
# =========================================================================
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))   # pre-normalised → cosine


def embed_subtasks(subtasks: List, embedder: Embedder) -> None:
    """Populate `subtask.embedding` in place for every subtask with a
    description. No-op for subtasks that already have an embedding."""
    pending = [(i, s) for i, s in enumerate(subtasks)
               if not getattr(s, "embedding", None)
               and getattr(s, "description", "")]
    if not pending:
        return
    texts = [s.description for _, s in pending]
    vecs = embedder.embed_batch(texts)
    for (_, s), v in zip(pending, vecs):
        s.embedding = list(v)


def dedupe_subtasks(subtasks: List, threshold: float = 0.92) -> List:
    """Purpose A — drop near-duplicate subtasks. When cosine > threshold,
    keep the one with higher importance (and on tie, earlier).

    Only same-modality subtasks can duplicate meaningfully — a doc_text and
    a video_visual subtask with similar descriptions are by design NOT
    duplicates (cross-modal verification pattern). So we only dedup within
    the same modality.

    Requires subtasks to have been embedded first. Untouched if not.
    """
    kept: List = []
    by_mod: dict = {}   # modality_tuple → list of kept indices
    for s in subtasks:
        emb = getattr(s, "embedding", None)
        if not emb:
            kept.append(s)
            continue
        mod_key = tuple(sorted(getattr(s, "modalities", []) or []))
        dup_idx = None
        for k_idx in by_mod.get(mod_key, []):
            other = kept[k_idx]
            if _cosine(emb, other.embedding) >= threshold:
                dup_idx = k_idx
                break
        if dup_idx is None:
            by_mod.setdefault(mod_key, []).append(len(kept))
            kept.append(s)
        else:
            # Merge policy: prefer higher importance, retain the kept
            # one's id (stability). If s is more important, upgrade.
            other = kept[dup_idx]
            if getattr(s, "importance", 0) > getattr(other, "importance", 0):
                other.importance = s.importance
                # Keep the fuller description of the two.
                if len(s.description) > len(other.description):
                    other.description = s.description
    return kept


def align_query_to_subtasks(query: str, subtasks: List,
                            embedder: Embedder,
                            top_k: int = 3) -> List:
    """Purpose B — rank subtasks by relevance to a given query string.
    Returns the top-K most similar subtasks. Interface wired but currently
    unused by the core loop; Round 3 retrieval tool may consume it.
    """
    if not subtasks:
        return []
    qvec = embedder.embed(query)
    scored = [(s, _cosine(qvec, s.embedding))
              for s in subtasks if getattr(s, "embedding", None)]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [s for s, _ in scored[:top_k]]
