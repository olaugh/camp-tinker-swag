"""Unwarp arched text to a straight strip so OCRs that struggle with curves
(Tesseract, plain CRNNs) can read it."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import Baseline


def unwarp_arc(img_bgr: np.ndarray, polygon: np.ndarray, baseline: Baseline,
               *, padding: float = 0.05, h_scale: float = 1.0) -> np.ndarray:
    """Resample a curved text region into a straight horizontal strip.

    Uses cv2.remap with polar coordinates around the baseline's circle.
    """
    if baseline.kind != "arc":
        x0, y0 = polygon.min(axis=0).astype(int)
        x1, y1 = polygon.max(axis=0).astype(int)
        H, W = img_bgr.shape[:2]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        return img_bgr[y0:y1, x0:x1].copy()

    cx, cy, r, _t0, _t1 = baseline.params
    # Determine the angular extent from the polygon itself (more reliable than fit endpoints).
    rel = polygon - np.array([cx, cy])
    angles = np.arctan2(rel[:, 1], rel[:, 0])
    radii  = np.sqrt(rel[:, 0]**2 + rel[:, 1]**2)
    # Pad
    a0, a1 = float(angles.min()), float(angles.max())
    # Avoid wrap (shouldn't happen for our 130° arcs but be safe)
    span = a1 - a0
    a0 -= span * padding * 0.1
    a1 += span * padding * 0.1
    r_inner = float(radii.min()) - (float(radii.max()) - float(radii.min())) * padding
    r_outer = float(radii.max()) + (float(radii.max()) - float(radii.min())) * padding
    r_inner = max(r_inner, 1.0)

    # Output dimensions
    out_h = max(int((r_outer - r_inner) * h_scale), 16)
    out_w = max(int((a1 - a0) * r * h_scale), 32)

    # Build mapping: for each output pixel (u, v),
    #   angle = a0 + (a1 - a0) * u / out_w
    #   radius = r_outer - (r_outer - r_inner) * v / out_h
    us = np.arange(out_w)
    vs = np.arange(out_h)
    aa = a0 + (a1 - a0) * us / max(out_w - 1, 1)
    rr = r_outer - (r_outer - r_inner) * vs / max(out_h - 1, 1)
    A, R = np.meshgrid(aa, rr)
    map_x = (cx + R * np.cos(A)).astype(np.float32)
    map_y = (cy + R * np.sin(A)).astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
