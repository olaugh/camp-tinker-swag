"""Intermediate representations passed between pipeline stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass
class Baseline:
    """Curve along which a piece of text is laid out."""
    kind: Literal["arc", "line"]
    # arc: (cx, cy, r, theta_start, theta_end). theta in radians, CCW from +x.
    # line: (x0, y0, x1, y1).
    params: tuple[float, ...]
    residual: float


@dataclass
class DetectedText:
    polygon: np.ndarray            # (N, 2), float, image coords (x, y)
    text: str                      # OCR result
    confidence: float              # detector or recognizer confidence
    baseline: Baseline | None = None
    crop: np.ndarray | None = None # tight RGB(A) crop covering the polygon
    # Per-character anchors detected from the input image: one dict per
    # non-space character, with keys "x", "y" (bbox-center anchor),
    # "rot_deg" (tangent rotation), "cap_h" (measured cap height). When
    # present, assemble emits per-letter <text> elements instead of
    # <textPath>.
    letter_anchors: list[dict] | None = None

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        xs, ys = self.polygon[:, 0], self.polygon[:, 1]
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


@dataclass
class FontMatch:
    family: str
    weight: int                    # CSS weight: 100..900
    style: Literal["normal", "italic"] = "normal"
    size_px: float = 0.0
    letter_spacing_em: float = 0.0
    dx: float = 0.0                # x offset applied to text origin
    dy: float = 0.0                # y offset
    score: float = float("inf")    # lower is better (L1 / (1-SSIM) composite)
    ssim: float = 0.0
    l1: float = float("inf")
    ttf_path: str = ""             # absolute path to the .ttf actually used


@dataclass
class TraceResult:
    svg_paths: list[str]           # raw <path d="..."/> attribute values
    width: int
    height: int


@dataclass
class PipelineResult:
    input_path: str
    output_svg_path: str
    texts: list[DetectedText] = field(default_factory=list)
    font_matches: list[FontMatch] = field(default_factory=list)
    trace: TraceResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    config: dict = field(default_factory=dict)
