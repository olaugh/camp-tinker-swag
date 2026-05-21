"""End-to-end pipeline runner."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import baseline as baseline_mod
from . import detect as detect_mod
from . import fontid as fontid_mod
from . import inpaint as inpaint_mod
from . import recognize as recognize_mod
from . import trace as trace_mod
from . import unwarp as unwarp_mod
from .assemble import assemble
from .types import DetectedText, FontMatch, PipelineResult


@dataclass
class PipelineConfig:
    detector: str = "synth"
    recognizer: str = "tesseract"
    inpainter: str = "white"
    tracer: str = "vtracer"
    font_corpus_dir: str = "fonts"
    text_height_factor: float = 1.4   # crop height multiplier vs polygon height
    detector_kwargs: dict[str, Any] = field(default_factory=dict)
    tracer_kwargs: dict[str, Any] = field(default_factory=dict)
    top_k_fonts: int = 5


def _crop_polygon(img: np.ndarray, polygon: np.ndarray,
                  *, height_mul: float = 1.0) -> np.ndarray:
    """Tight bbox crop. (For arched text we crop the bbox of the polygon.)"""
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    H, W = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = (y1 - y0) * height_mul
    w = (x1 - x0)
    x0 = int(max(0, cx - w/2))
    x1 = int(min(W, cx + w/2))
    y0 = int(max(0, cy - h/2))
    y1 = int(min(H, cy + h/2))
    return img[y0:y1, x0:x1].copy()


def run(input_path: Path | str,
        output_path: Path | str,
        cfg: PipelineConfig,
        *,
        badge=None) -> PipelineResult:
    """Run a full configuration end-to-end. Returns a PipelineResult."""
    timings: dict[str, float] = {}
    config_dump = {
        "detector": cfg.detector, "recognizer": cfg.recognizer,
        "inpainter": cfg.inpainter, "tracer": cfg.tracer,
        "detector_kwargs": cfg.detector_kwargs,
        "tracer_kwargs": cfg.tracer_kwargs,
    }
    out = PipelineResult(input_path=str(input_path), output_svg_path=str(output_path),
                         timings=timings, config=config_dump)

    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(input_path)
    H, W = img_bgr.shape[:2]
    timings["load"] = time.perf_counter() - t0

    # 1. detect
    t0 = time.perf_counter()
    det_fn = detect_mod.REGISTRY[cfg.detector]
    kw = dict(cfg.detector_kwargs)
    if cfg.detector == "synth":
        kw["badge"] = badge
    polygons = det_fn(img_bgr, **kw)
    timings["detect"] = time.perf_counter() - t0

    # 2. recognize + baseline + crop, per polygon
    t0 = time.perf_counter()
    rec_fn = recognize_mod.REGISTRY[cfg.recognizer]
    texts: list[DetectedText] = []
    for poly in polygons:
        bl = baseline_mod.fit(poly)
        # Unwarp arched text BEFORE recognition; straight text just gets cropped.
        if bl.kind == "arc":
            crop = unwarp_mod.unwarp_arc(img_bgr, poly, bl)
        else:
            crop = _crop_polygon(img_bgr, poly, height_mul=cfg.text_height_factor)
        if crop.size == 0:
            continue
        text, conf = rec_fn(crop[:, :, ::-1])
        clean = "".join(c for c in text if c.isalnum() or c.isspace()).strip().upper()
        dt = DetectedText(polygon=poly, text=clean, confidence=conf,
                          baseline=bl, crop=crop)
        texts.append(dt)
    timings["recognize+baseline"] = time.perf_counter() - t0
    out.texts = texts

    # 3. font id
    t0 = time.perf_counter()
    corpus = fontid_mod.discover_corpus(cfg.font_corpus_dir)
    matches: list[FontMatch] = []
    for dt in texts:
        if not dt.text or not corpus:
            matches.append(FontMatch(family="sans-serif", weight=700))
            continue
        gray = cv2.cvtColor(dt.crop, cv2.COLOR_BGR2GRAY)
        candidates = fontid_mod.match(gray, dt.text, corpus, top_k=cfg.top_k_fonts)
        best = candidates[0] if candidates else FontMatch(family="sans-serif", weight=700)
        # Re-record a sensible size for assembly.
        # For arched text the polygon's y-extent includes arc sag; use the
        # unwarped strip height instead. For straight text use the polygon h.
        if dt.baseline and dt.baseline.kind == "arc":
            phys_h = float(dt.crop.shape[0])
        else:
            phys_h = float(dt.polygon[:, 1].max() - dt.polygon[:, 1].min())
        best_dict = best.__dict__.copy()
        # font-size for SVG text usually >= ink height; bump slightly for cap-letters.
        best_dict["size_px"] = phys_h * 1.4
        matches.append(FontMatch(**best_dict))
    timings["fontid"] = time.perf_counter() - t0
    out.font_matches = matches

    # 4. inpaint
    t0 = time.perf_counter()
    inp_fn = inpaint_mod.REGISTRY[cfg.inpainter]
    mask = inpaint_mod.polygons_to_mask([dt.polygon for dt in texts], (H, W), inflate_px=4)
    residual = inp_fn(img_bgr, mask)
    timings["inpaint"] = time.perf_counter() - t0

    # 5. trace
    t0 = time.perf_counter()
    tr_fn = trace_mod.REGISTRY[cfg.tracer]
    trace = tr_fn(residual, **cfg.tracer_kwargs)
    timings["trace"] = time.perf_counter() - t0
    out.trace = trace

    # 6. assemble
    t0 = time.perf_counter()
    assemble(trace, texts, matches, output_path=output_path)
    timings["assemble"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    return out
