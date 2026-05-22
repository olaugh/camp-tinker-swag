"""Fit a circular arc or a straight line through the centerline of a text polygon."""
from __future__ import annotations

import math

import numpy as np
from skimage.measure import CircleModel, LineModelND, ransac

from .types import Baseline


def _centerline(polygon: np.ndarray) -> np.ndarray:
    """Split polygon vertices into 'top' and 'bottom' halves and take midpoints.

    The polygon is assumed to be roughly convex with an even count, half along
    one long edge and half along the other (which is the typical detector output).
    """
    pts = np.asarray(polygon, dtype=np.float64)
    n = len(pts)
    if n < 4:
        return pts
    if n % 2 == 0:
        # Detector polygons usually go: top edge L->R, then bottom edge R->L
        half = n // 2
        top = pts[:half]
        bot = pts[half:][::-1]
        m = min(len(top), len(bot))
        return (top[:m] + bot[:m]) / 2
    # Otherwise: sort by x, then pair points by x-rank.
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    half = n // 2
    top = pts[:half]
    bot = pts[half:][::-1]
    m = min(len(top), len(bot))
    return (top[:m] + bot[:m]) / 2


def fit(polygon: np.ndarray, *, seed: int = 0) -> Baseline:
    """Choose circle or line based on which has lower residual."""
    center_pts = _centerline(polygon)
    if len(center_pts) < 3:
        x0, y0 = polygon[0]
        x1, y1 = polygon[-1]
        return Baseline(kind="line", params=(float(x0), float(y0), float(x1), float(y1)), residual=0.0)

    # Circle fit
    try:
        np.random.seed(seed)
        cm = CircleModel.from_estimate(center_pts)
        cx, cy = cm.center
        r = cm.radius
        dists = np.sqrt((center_pts[:, 0] - cx)**2 + (center_pts[:, 1] - cy)**2)
        circ_residual = float(np.mean(np.abs(dists - r)))
    except Exception:
        cx = cy = r = 0.0
        circ_residual = float("inf")

    # Line fit
    try:
        lm = LineModelND.from_estimate(center_pts)
        line_residual = float(np.mean(np.abs(lm.residuals(center_pts))))
        origin, direction = lm.origin, lm.direction
        proj = (center_pts - origin) @ direction
        t0, t1 = proj.min(), proj.max()
        x0, y0 = origin + t0 * direction
        x1, y1 = origin + t1 * direction
    except Exception:
        x0 = y0 = x1 = y1 = 0.0
        line_residual = float("inf")

    # Pick the one with smaller residual *relative to the polygon scale*.
    if circ_residual < line_residual * 0.9 and r < 10 * np.ptp(center_pts):
        # angular endpoints
        t0 = math.atan2(center_pts[0, 1] - cy, center_pts[0, 0] - cx)
        t1 = math.atan2(center_pts[-1, 1] - cy, center_pts[-1, 0] - cx)
        return Baseline(kind="arc",
                        params=(float(cx), float(cy), float(r), float(t0), float(t1)),
                        residual=circ_residual)
    return Baseline(kind="line",
                    params=(float(x0), float(y0), float(x1), float(y1)),
                    residual=line_residual)
