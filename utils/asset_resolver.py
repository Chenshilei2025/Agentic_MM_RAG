"""Resolve candidate asset references into injectable content blocks.

The Decision Agent's INSPECT_EVIDENCE command requests specific candidate
ids. The Orchestrator resolves those ids → raw assets (images, text blocks,
video clip URLs) → multimodal content blocks suitable for injection into
the DA's next prompt.

This module owns that resolution. Abstract interface so you can plug in:
  - LocalFileResolver   default. reads file:// URIs + reads text_range from
                        doc chunks' content field
  - S3Resolver          (not implemented; stub)
  - FrameExtractorResolver (not implemented; would call ffmpeg)

Content block format is Anthropic-style (text/image blocks), which is also
compatible with OpenAI-style multimodal once we translate on provider side.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import base64, os


@dataclass
class ContentBlock:
    """One injectable block. Either 'text' or 'image'.
    For image: base64 data + media_type. For text: plain string."""
    type: str                       # "text" | "image"
    text: Optional[str] = None
    image_data: Optional[str] = None    # base64
    media_type: Optional[str] = None    # e.g. "image/jpeg"
    source_label: str = ""          # human-readable provenance

    def to_anthropic(self) -> Dict[str, Any]:
        if self.type == "text":
            return {"type": "text", "text": self.text}
        return {"type": "image",
                "source": {"type": "base64",
                           "media_type": self.media_type,
                           "data": self.image_data}}


class AssetResolver:
    """Abstract base. Subclasses implement resolve_one()."""

    def resolve_one(self, candidate: Dict[str, Any]) -> List[ContentBlock]:
        raise NotImplementedError

    def resolve_many(self, candidates: List[Dict[str, Any]],
                     header: str = "") -> List[ContentBlock]:
        out: List[ContentBlock] = []
        if header:
            out.append(ContentBlock(type="text", text=header))
        for c in candidates:
            blocks = self.resolve_one(c)
            out.extend(blocks)
        return out


class LocalFileResolver(AssetResolver):
    """Default resolver.
    - asset_type='image' + asset_uri='file://…' → read + base64
    - asset_type absent → treat candidate.content as text block
    """

    # Cap image size crudely (prevents giant-image token explosions). A
    # production system would use Pillow to resize; here we just reject.
    MAX_IMAGE_BYTES = 3 * 1024 * 1024

    def __init__(self, resize_to_px: Optional[int] = 1024):
        self.resize_to_px = resize_to_px

    def resolve_one(self, candidate: Dict[str, Any]) -> List[ContentBlock]:
        meta = candidate.get("meta") or {}
        asset_type = meta.get("asset_type")
        label = self._label(candidate, meta)

        if asset_type == "image":
            uri = meta.get("asset_uri", "")
            if not uri.startswith("file://"):
                # Unsupported scheme — degrade gracefully to caption text.
                return [ContentBlock(
                    type="text",
                    text=f"[image at {uri} unavailable; caption: "
                         f"{candidate.get('content','(none)')}]",
                    source_label=label)]
            path = uri[len("file://"):]
            if not os.path.exists(path):
                return [ContentBlock(
                    type="text",
                    text=f"[image path {path} not found; caption: "
                         f"{candidate.get('content','(none)')}]",
                    source_label=label)]
            size = os.path.getsize(path)
            if size > self.MAX_IMAGE_BYTES:
                return [ContentBlock(
                    type="text",
                    text=f"[image {path} too large ({size} bytes); "
                         f"caption: {candidate.get('content','(none)')}]",
                    source_label=label)]
            try:
                raw = open(path, "rb").read()
                b64 = base64.b64encode(raw).decode("ascii")
                media = self._media_type_from_ext(path)
                # Two blocks per image: a caption header + the image itself.
                return [
                    ContentBlock(type="text",
                                 text=f"\n--- {label} ---",
                                 source_label=label),
                    ContentBlock(type="image",
                                 image_data=b64,
                                 media_type=media,
                                 source_label=label),
                ]
            except OSError as e:
                return [ContentBlock(
                    type="text",
                    text=f"[image load failed {e}; caption: "
                         f"{candidate.get('content','(none)')}]",
                    source_label=label)]

        # Default: text block. Works for doc_text / video_text candidates.
        # We include source provenance so DA can track lineage.
        text = candidate.get("content", "") or "(empty)"
        return [ContentBlock(
            type="text",
            text=f"\n--- {label} ---\n{text}",
            source_label=label)]

    # -- helpers --
    def _label(self, cand, meta) -> str:
        id_ = cand.get("id", "?")
        bits = [f"id={id_}"]
        for k in ("source", "page", "t_start", "t", "frame_idx"):
            if k in meta:
                bits.append(f"{k}={meta[k]}")
        return " ".join(bits)

    def _media_type_from_ext(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".gif":  "image/gif",
            ".webp": "image/webp",
        }.get(ext, "image/jpeg")
