"""Text-detection adapters returning polygon vertices.

Every adapter has the signature:
    detect(img_bgr: np.ndarray) -> list[np.ndarray]   # each (N, 2) polygon, in pixel coords
"""
from __future__ import annotations

import importlib.util
from typing import Callable

import numpy as np


Detector = Callable[[np.ndarray], list[np.ndarray]]


def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


# -- geometry-driven detection for camp-style badges ---------------------- #

def detect_camp_geom(img_bgr: np.ndarray,
                     *, arc_span_deg: float = 140.0,
                     arc_outside_ratio: float = 0.10,
                     title_height_ratio: float = 0.075,
                     year_y_ratio: float = 0.08,
                     year_height_ratio: float = 0.07) -> list[np.ndarray]:
    """Camp-Tinker-style polygons derived from the badge's outer ring.

    Use this when tesseract/easyocr miss arched text on a logo we already
    understand the layout of. We locate the badge with Hough, then build
    a polygon ringing the outer circle for the title, and an axis-aligned
    rectangle below the badge for the year. Ratios are relative to the
    image's short edge so they scale with input resolution.

    Returns [title_arc_polygon, year_rect_polygon].
    """
    import math
    from . import geom as geom_mod

    badge = geom_mod.find_badge_center(img_bgr)
    if badge is None:
        return []
    cx, cy, r_outer = badge
    H, W = img_bgr.shape[:2]
    short = float(min(H, W))

    # Title sits on an arc just OUTSIDE the outer ring.
    title_h = short * title_height_ratio
    arc_r = r_outer + short * arc_outside_ratio
    half_h = title_h * 0.6
    arc_span = math.radians(arc_span_deg)
    t0, t1 = -math.pi / 2 - arc_span / 2, -math.pi / 2 + arc_span / 2
    n = 24
    upper, lower = [], []
    for i in range(n + 1):
        t = t0 + (t1 - t0) * i / n
        upper.append((cx + (arc_r + half_h) * math.cos(t),
                      cy + (arc_r + half_h) * math.sin(t)))
        lower.append((cx + (arc_r - half_h) * math.cos(t),
                      cy + (arc_r - half_h) * math.sin(t)))
    title_poly = np.array(upper + lower[::-1], dtype=np.float32)

    # Year sits below the outer ring, centered on cx.
    year_y = cy + r_outer + short * year_y_ratio
    year_h = short * year_height_ratio
    year_w = year_h * 1.8  # ~4 digits at ~0.5em width
    year_poly = np.array([
        [cx - year_w, year_y - year_h * 1.2],
        [cx + year_w, year_y - year_h * 1.2],
        [cx + year_w, year_y + year_h * 0.4],
        [cx - year_w, year_y + year_h * 0.4],
    ], dtype=np.float32)
    return [title_poly, year_poly]


# -- synthetic ground truth (uses synth.py geometry) ----------------------- #

def detect_synth(img_bgr: np.ndarray, badge=None) -> list[np.ndarray]:
    """Return exact polygons from a SynthBadge -- only works on our synthesizer's output."""
    import math
    if badge is None:
        raise ValueError("detect_synth requires the SynthBadge instance")
    H, W = img_bgr.shape[:2]
    cx, cy = badge.arc_center
    r = badge.arc_radius
    half_h = badge.title_px * 0.6   # approx text height; tune
    arc_span = math.radians(140)
    theta0, theta1 = -math.pi/2 - arc_span/2, -math.pi/2 + arc_span/2
    n = 24
    upper, lower = [], []
    for i in range(n + 1):
        t = theta0 + (theta1 - theta0) * i / n
        upper.append((cx + (r + half_h) * math.cos(t), cy + (r + half_h) * math.sin(t)))
        lower.append((cx + (r - half_h) * math.cos(t), cy + (r - half_h) * math.sin(t)))
    arc_poly = np.array(upper + lower[::-1], dtype=np.float32)

    yx, yy = badge.year_pos
    h2 = badge.year_px * 0.65
    w2 = badge.year_px * 1.8  # 4 digits ~= 4 * 0.5em width
    year_poly = np.array([
        [yx - w2, yy - h2 * 1.2],
        [yx + w2, yy - h2 * 1.2],
        [yx + w2, yy + h2 * 0.4],
        [yx - w2, yy + h2 * 0.4],
    ], dtype=np.float32)
    return [arc_poly, year_poly]


# -- Tesseract via pytesseract --------------------------------------------- #

def detect_tesseract(img_bgr: np.ndarray) -> list[np.ndarray]:
    import pytesseract
    from PIL import Image
    pil = Image.fromarray(img_bgr[:, :, ::-1])  # BGR->RGB
    # Use word-level boxes. Tesseract on arched text will return per-word fragments
    # along the arc -- we'll merge them by polygon hull below.
    data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT,
                                     config="--psm 11")  # sparse text
    polys: list[np.ndarray] = []
    for i, conf in enumerate(data["conf"]):
        try:
            c = float(conf)
        except (TypeError, ValueError):
            continue
        if c < 30:
            continue
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        polys.append(np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32))
    return polys


# -- EasyOCR --------------------------------------------------------------- #

def detect_easyocr(img_bgr: np.ndarray, reader=None) -> list[np.ndarray]:
    import easyocr
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    rgb = img_bgr[:, :, ::-1].copy()
    # detail=1 returns [bbox, text, conf]; bbox is 4 corner points.
    results = reader.readtext(rgb, detail=1, paragraph=False)
    polys: list[np.ndarray] = []
    for bbox, _text, conf in results:
        if float(conf) < 0.3:
            continue
        polys.append(np.array(bbox, dtype=np.float32))
    return polys


# -- docTR ----------------------------------------------------------------- #

def detect_doctr(img_bgr: np.ndarray, predictor=None) -> list[np.ndarray]:
    import torch
    from doctr.models import ocr_predictor
    if predictor is None:
        predictor = ocr_predictor(det_arch="db_resnet50", reco_arch="crnn_vgg16_bn",
                                  pretrained=True, assume_straight_pages=False)
    H, W = img_bgr.shape[:2]
    rgb = img_bgr[:, :, ::-1].copy()
    out = predictor([rgb])
    polys: list[np.ndarray] = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    g = word.geometry
                    # geometry is a tuple of (x_min,y_min)(x_max,y_max) for straight,
                    # or 4 points for non-straight. Normalize either way.
                    if isinstance(g[0], (list, tuple)) and len(g) == 4:
                        pts = np.array(g, dtype=np.float32)
                    else:
                        (x0, y0), (x1, y1) = g
                        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                    pts[:, 0] *= W
                    pts[:, 1] *= H
                    polys.append(pts)
    return polys


# -- registry -------------------------------------------------------------- #

REGISTRY: dict[str, Detector] = {
    "synth":     detect_synth,
    "camp_geom": detect_camp_geom,
    "tesseract": detect_tesseract,
    "easyocr":   detect_easyocr,
    "doctr":     detect_doctr,
}
