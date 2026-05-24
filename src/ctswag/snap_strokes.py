"""Snap vtracer-output paths to axis-aligned strokes where the source's
intent is clearly axis-aligned (e.g. waterfall ticks, tree trunks).

vtracer emits each ink region as a filled cubic-Bezier contour. For a
nominally-vertical stroke a few degrees off-axis the contour comes out
visibly tilted; we replace that contour with an exactly-vertical
`<line>` of matching width.

Strictly scoped:
  * Only paths whose bbox center lies BELOW a detected horizon line (the
    mountain baseline in a badge) are considered. Above that line is
    sun rays, mountains, etc. -- we don't want to touch those.
  * Only "tall-narrow" paths (aspect ratio >= 2) qualify -- avoids
    snapping square/round shapes.
  * Only paths whose PCA major-axis is within `tolerance_deg` of vertical
    get snapped. A snowcap squiggle or a mountain ridge stays as-is
    because its principal axis isn't near-vertical.
"""
from __future__ import annotations

import re
from typing import Iterable

import cv2
import numpy as np


_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")
_D_RE = re.compile(r'd="([^"]*)"')
_TRANSLATE_RE = re.compile(r'translate\(\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*\)')


def _translate_offset(path_str: str) -> tuple[float, float]:
    """Parse a translate(x, y) from the path's transform attribute, or (0, 0)
    if the path has no transform. vtracer emits each shape at local coords
    starting near origin and positions it with translate(); without
    applying it, every shape's bbox looks like it's at (0, 0)."""
    m = _TRANSLATE_RE.search(path_str)
    if not m:
        return 0.0, 0.0
    return float(m.group(1)), float(m.group(2))


def detect_horizon_y(gray_residual: np.ndarray,
                     cx: float, cy: float, inner_r: float,
                     *, max_angle_deg: float = 3.0,
                     min_length_ratio: float = 0.5
                     ) -> float | None:
    """Find the y-coordinate of the most prominent horizontal line in the
    inside-the-rings region below the badge center.

    Returns the y value of the longest near-horizontal segment, or None
    if no qualifying line is found.
    """
    H, W = gray_residual.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((np.hypot(xx - cx, yy - cy) < inner_r) & (yy > cy)).astype(np.uint8)
    masked = np.where(mask > 0, gray_residual, np.uint8(255))
    edges = cv2.Canny(masked, 50, 150)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180.0,
        threshold=80,
        minLineLength=int(inner_r * min_length_ratio),
        maxLineGap=20,
    )
    if lines is None:
        return None
    best_len = 0.0
    best_y: float | None = None
    cos_tol = np.cos(np.deg2rad(max_angle_deg))
    for line in lines:
        x1, y1, x2, y2 = (float(v) for v in line[0])
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length <= 0:
            continue
        # cos(angle from horizontal) = |dx| / length
        if abs(dx) / length < cos_tol:
            continue
        if length > best_len:
            best_len = length
            best_y = (y1 + y2) / 2.0
    return best_y


def _path_points(d: str) -> np.ndarray:
    """Parse an SVG path `d` attribute and return all (x, y) control points
    as an (N, 2) ndarray. Doesn't preserve command context -- good enough
    for bbox / PCA on convex-ish filled contours from vtracer.
    """
    nums = [float(s) for s in _NUM_RE.findall(d)]
    if len(nums) < 4:
        return np.empty((0, 2), dtype=np.float64)
    n = len(nums) // 2
    return np.asarray(
        [(nums[2 * i], nums[2 * i + 1]) for i in range(n)],
        dtype=np.float64,
    )


def _pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (centroid, eigvals_asc, eigvecs) for the (N, 2) array `pts`."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    return centroid, eigvals, eigvecs


def _trim_to_body(pts: np.ndarray, trim_ratio: float = 0.2) -> np.ndarray:
    """Drop the top/bottom `trim_ratio` of points along the path's major
    axis. Excludes vtracer's rounded-tip cap pixels from a subsequent PCA
    angle measurement so that what's left is the straight body of the
    stroke. Caller still gets the full bbox from the untrimmed points.
    """
    if len(pts) < 4:
        return pts
    centroid, _eigvals, eigvecs = _pca(pts)
    major = eigvecs[:, -1]
    proj = (pts - centroid) @ major
    t_min, t_max = float(proj.min()), float(proj.max())
    extent = t_max - t_min
    if extent <= 0:
        return pts
    trim = extent * trim_ratio
    keep = (proj >= t_min + trim) & (proj <= t_max - trim)
    if keep.sum() < 4:
        return pts
    return pts[keep]


def snap_vertical_below_horizon(
        svg_paths: Iterable[str],
        horizon_y: float,
        *, tolerance_deg: float = 8.0,
        min_aspect_ratio: float = 2.0,
        min_height_px: float = 8.0,
        max_width_px: float = 40.0,
        max_eigval_ratio: float = 0.7,
        tip_trim_ratio: float = 0.2,
        debug_records: list[dict] | None = None,
        ) -> list[str]:
    """For each `<path d="..."/>` string in `svg_paths`, replace it with an
    axis-aligned `<line>` element if it qualifies as a near-vertical stroke
    below the horizon. Otherwise pass it through unchanged.

    A path qualifies when ALL of these are true:
      * Its bbox center y > `horizon_y` (below the mountain baseline).
      * height / width >= `min_aspect_ratio` AND height >= `min_height_px`
        (genuinely tall-narrow, not a square blob).
      * width <= `max_width_px` (excludes shapes whose bbox is wider than
        a normal stroke -- a tree trunk's bbox includes its branches).
      * eigvals[minor] / eigvals[major] <= `max_eigval_ratio` (pixels
        actually cluster tightly along a line, not a tree-with-branches
        shape that PCA still calls vertical because the trunk dominates).
      * PCA major-axis is within `tolerance_deg` of vertical.
    """
    sin_tol = float(np.sin(np.deg2rad(tolerance_deg)))
    out: list[str] = []

    def _record(bbox=None, angle_deg=None, eigval_ratio=None,
                decision="snapped", reason=""):
        if debug_records is None:
            return
        debug_records.append({
            "bbox": bbox,
            "angle_deg": angle_deg,
            "eigval_ratio": eigval_ratio,
            "decision": decision,
            "reason": reason,
        })

    for path_str in svg_paths:
        m = _D_RE.search(path_str)
        if not m:
            out.append(path_str)
            _record(decision="skip", reason="no d attr")
            continue
        local_pts = _path_points(m.group(1))
        if len(local_pts) < 4:
            out.append(path_str)
            _record(decision="skip", reason="too few points")
            continue
        # Apply translate(x, y) from the path's transform attribute so we
        # work in world coords -- vtracer emits each path with local coords
        # near origin and a translate() to position it.
        tx, ty = _translate_offset(path_str)
        full_pts = local_pts + np.array([tx, ty])
        full_y_max = float(full_pts[:, 1].max())
        # Drop paths entirely above the horizon (they're sun rays /
        # mountain ridges / etc.). For paths that STRADDLE the horizon
        # (typically a waterfall tick that vtracer merged with the
        # mountain base into one L-shape), keep only the below-horizon
        # portion of the points -- that subset is the tick alone, and
        # the size/angle filters below should now accept it.
        if full_y_max <= horizon_y:
            out.append(path_str)
            _record(decision="reject_above", reason="entirely above horizon")
            continue
        below_mask = full_pts[:, 1] > horizon_y
        if below_mask.sum() >= 4:
            pts = full_pts[below_mask]
        else:
            pts = full_pts
        x_min, y_min = float(pts[:, 0].min()), float(pts[:, 1].min())
        x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
        w = x_max - x_min
        h = y_max - y_min
        bbox = (x_min, y_min, x_max, y_max)
        if h < min_height_px or w <= 0 or h / w < min_aspect_ratio:
            out.append(path_str)
            _record(bbox=bbox, decision="reject_shape",
                    reason=f"h/w={h/(w or 1):.1f} too low or h<{min_height_px}")
            continue
        if w > max_width_px:
            out.append(path_str)
            _record(bbox=bbox, decision="reject_wide",
                    reason=f"w={w:.0f} > {max_width_px}")
            continue
        # `pts` now only contains the below-horizon subset (when the path
        # straddles the horizon) or the full path (when it's entirely
        # below). Either way we don't need the redundant cy > horizon
        # check here.
        # PCA on the FULL path. The rounded-tip cap pixels inflate the
        # minor/major ratio (a rounded rectangle outline measures
        # ~0.05-0.15 vs ~0.01 for a uniformly-filled rectangle), so
        # `max_eigval_ratio` is set to admit those values. Trees with
        # branches are well above (~0.2+) and are already excluded by
        # `max_width_px` since their bbox includes branches.
        _full_centroid, full_eigvals, full_eigvecs = _pca(pts)
        if full_eigvals[1] < 1e-9:
            out.append(path_str)
            _record(bbox=bbox, decision="reject_degenerate", reason="eigvals=0")
            continue
        eigval_ratio = float(full_eigvals[0] / full_eigvals[1])
        axis = full_eigvecs[:, -1]
        norm = float(np.hypot(axis[0], axis[1]))
        angle_from_vertical_deg = (float(np.degrees(np.arctan2(abs(axis[0]), abs(axis[1]))))
                                    if norm > 1e-9 else 90.0)
        if eigval_ratio > max_eigval_ratio:
            out.append(path_str)
            _record(bbox=bbox, angle_deg=angle_from_vertical_deg,
                    eigval_ratio=eigval_ratio, decision="reject_ratio",
                    reason=f"eigval_ratio={eigval_ratio:.2f} > {max_eigval_ratio}")
            continue
        if norm < 1e-9:
            out.append(path_str)
            continue
        # Vertical = (0, ±1). Distance from vertical = |dx| / norm.
        if abs(axis[0]) / norm > sin_tol:
            out.append(path_str)
            _record(bbox=bbox, angle_deg=angle_from_vertical_deg,
                    eigval_ratio=eigval_ratio, decision="reject_angle",
                    reason=f"angle={angle_from_vertical_deg:.1f}° > {tolerance_deg}°")
            continue
        # Snap. Use bbox-center x and bbox y extent. Stroke width = bbox w.
        cx_line = (x_min + x_max) / 2.0
        out.append(
            f'<line x1="{cx_line:.2f}" y1="{y_min:.2f}" '
            f'x2="{cx_line:.2f}" y2="{y_max:.2f}" '
            f'stroke="black" stroke-width="{w:.2f}" fill="none" '
            f'stroke-linecap="round"/>'
        )
        _record(bbox=bbox, angle_deg=angle_from_vertical_deg,
                eigval_ratio=eigval_ratio, decision="snapped",
                reason="ok")
    return out


def render_debug_overlay(
        debug_records: list[dict],
        backdrop: np.ndarray,
        horizon_y: float | None = None,
        ) -> np.ndarray:
    """Render each path-candidate's bbox onto a copy of `backdrop`, color-
    coded by decision. snapped=green, reject_ratio=orange,
    reject_angle=red, reject_wide / reject_shape / reject_above = gray.
    Annotates each box with angle (degrees from vertical) and eigval ratio.
    Useful for understanding which paths the snap is firing on and why
    the rest are being rejected.
    """
    if backdrop.ndim == 2:
        viz = cv2.cvtColor(backdrop, cv2.COLOR_GRAY2BGR)
    else:
        viz = backdrop.copy()
    # Dim background so annotations stand out.
    viz = (viz.astype(np.int16) // 2 + 128).astype(np.uint8)
    H, W = viz.shape[:2]

    if horizon_y is not None:
        cv2.line(viz, (0, int(horizon_y)), (W - 1, int(horizon_y)),
                 (180, 0, 180), 2)

    decision_color = {
        "snapped":           (0, 200,   0),     # green
        "reject_ratio":      (0, 165, 255),     # orange
        "reject_angle":      (0,   0, 255),     # red
        "reject_above":      (150, 150, 150),   # gray
        "reject_shape":      (180, 180, 180),
        "reject_wide":       (200, 200, 200),
        "reject_degenerate": (220, 220, 220),
        "skip":              (240, 240, 240),
    }
    for rec in debug_records:
        bbox = rec.get("bbox")
        if bbox is None:
            continue
        x_min, y_min, x_max, y_max = bbox
        color = decision_color.get(rec["decision"], (255, 255, 255))
        cv2.rectangle(viz, (int(x_min) - 1, int(y_min) - 1),
                      (int(x_max) + 1, int(y_max) + 1), color, 2)
        label_parts = []
        if rec.get("angle_deg") is not None:
            label_parts.append(f"{rec['angle_deg']:.1f}°")
        if rec.get("eigval_ratio") is not None:
            label_parts.append(f"r={rec['eigval_ratio']:.2f}")
        label = " ".join(label_parts)
        if label:
            tx = min(int(x_max) + 4, W - 80)
            ty = int((y_min + y_max) / 2)
            cv2.putText(viz, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                        cv2.LINE_AA)
    return viz
