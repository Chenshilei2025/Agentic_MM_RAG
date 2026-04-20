"""FAISS adapter for offline-vectorized indices.

Drop-in replacement for VectorIndex when you have real embeddings on disk.
Lazily imports faiss so the base system runs without it installed.

EXPECTED OFFLINE ARTIFACTS (per modality):
    <root>/<modality>/index.faiss       FAISS index (IndexFlatIP or IVF_PQ)
    <root>/<modality>/meta.jsonl        one JSON object per vector, in order:
                                        {"id": "...", "content": "...",
                                         "meta": {...}}
Encoder:
    A callable (query:str) -> np.ndarray[d] returning a L2-normalized vector.
    Provide one per modality — typically a text encoder for *_text indices
    and a vision encoder for *_visual indices (after projecting the text
    query into the shared multimodal embedding space).

USAGE:
    from memory.faiss_adapter import FaissIndex
    from memory.store import MultiIndexStore
    store = MultiIndexStore({
        "doc_text":     FaissIndex("doc_text",     "/data/idx", text_encoder),
        "doc_visual":   FaissIndex("doc_visual",   "/data/idx", clip_text_enc),
        "video_text":   FaissIndex("video_text",   "/data/idx", text_encoder),
        "video_visual": FaissIndex("video_visual", "/data/idx", clip_text_enc),
    })
"""
from typing import Any, Callable, Dict, List
import json, os

MODALITIES = ("doc_text", "doc_visual", "video_text", "video_visual")


class FaissIndex:
    """Thin adapter around a prebuilt FAISS index + JSONL metadata file."""

    def __init__(self, modality: str, root_dir: str,
                 encoder: Callable[[str], "np.ndarray"],
                 normalize: bool = True):
        assert modality in MODALITIES, f"unknown modality: {modality}"
        self.modality = modality
        self.root_dir = root_dir
        self.encoder = encoder
        self.normalize = normalize
        self._index = None
        self._meta: List[Dict[str, Any]] = []

    def _ensure_loaded(self):
        if self._index is not None:
            return
        try:
            import faiss  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "faiss is required for FaissIndex. "
                "Install with: pip install faiss-cpu (or faiss-gpu)"
            ) from e
        idx_path = os.path.join(self.root_dir, self.modality, "index.faiss")
        meta_path = os.path.join(self.root_dir, self.modality, "meta.jsonl")
        if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(
                f"missing artifacts for {self.modality}: {idx_path} / {meta_path}")
        self._index = faiss.read_index(idx_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            self._meta = [json.loads(line) for line in f if line.strip()]
        if self._index.ntotal != len(self._meta):
            raise ValueError(
                f"{self.modality}: index has {self._index.ntotal} vectors "
                f"but meta has {len(self._meta)} rows")

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        import numpy as np
        vec = self.encoder(query).astype("float32").reshape(1, -1)
        if self.normalize:
            n = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12
            vec = vec / n
        D, I = self._index.search(vec, k)
        out: List[Dict[str, Any]] = []
        for rank, (score, idx) in enumerate(zip(D[0].tolist(), I[0].tolist())):
            if idx < 0 or idx >= len(self._meta):
                continue
            row = self._meta[idx]
            out.append({
                "id":       row.get("id", str(idx)),
                "content":  row.get("content", ""),
                "meta":     row.get("meta", {}),
                "modality": self.modality,
                "score":    float(score),
                "rank":     rank,
            })
        return out
