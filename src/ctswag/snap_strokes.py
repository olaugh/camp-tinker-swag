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


def _emit_snapped_line(x_min: float, x_max: float, y_min: float, y_max: float,
                       inner_circle: tuple[float, float, float] | None,
                       ring_join_px: float,
                       *, top_flat: bool = False) -> str:
    """Render a single snapped vertical tick into SVG markup.
    Handles cap inset (so round caps don't overshoot the original tip
    centres) and the flat-cap-at-ring-junction case. When `top_flat` is
    True (e.g. for comb teeth that attach to the horizon line), don't inset
    the top — let it butt directly against the line above with no round cap."""
    # Subpixel sweep on the asset showed that REDUCING the snap-line
    # stroke by ~1.5 px improves inside-ring SSIM (resvg renders the
    # raw bbox width slightly too heavy because the mask-derived width
    # includes a half-pixel of AA halo on each side).
    w = (x_max - x_min) - 1.5
    if w <= 0:
        w = x_max - x_min  # fallback for very-thin ticks
    cx = (x_min + x_max) / 2.0
    cap_inset = w / 2.0
    if top_flat:
        # Top sits flush at y_min (no cap above); only bottom may be capped.
        y1 = y_min
    else:
        y1 = y_min + cap_inset
    y2 = y_max - cap_inset
    if y1 > y2:
        y1 = y2 = (y_min + y_max) / 2.0
    cap_bottom = "round"
    if inner_circle is not None:
        icx, icy, ir = inner_circle
        dx = cx - icx
        if abs(dx) < ir:
            ring_y_at_x = icy + float(np.sqrt(ir * ir - dx * dx))
            bottom_visible = y2 + cap_inset
            if abs(ring_y_at_x - bottom_visible) <= ring_join_px:
                cap_bottom = "butt"
                # End the snap line right at the ring centerline. The ring
                # is drawn on top of geometry, so its stroke half-covers
                # the line's bottom — meaning the line visually butts
                # against the ring's inner edge with no rounded cap and
                # no visible protrusion past the ring's outer edge.
                y2 = ring_y_at_x
    # Build the body line with the appropriate linecap.
    if top_flat and cap_bottom == "butt":
        linecap = "butt"
        body = (f'<line x1="{cx:.2f}" y1="{y1:.2f}" '
                f'x2="{cx:.2f}" y2="{y2:.2f}" '
                f'stroke="black" stroke-width="{w:.2f}" fill="none" '
                f'stroke-linecap="butt"/>')
        return body
    if top_flat:
        # Top flat, bottom round: butt cap line + circle at the bottom.
        # Wrap in <g> so the assemble step's etree.fromstring sees a single
        # root element (without the wrapper it falls back to escaping the
        # whole snippet into a `d` attribute, which doesn't render).
        return (
            f'<g>'
            f'<line x1="{cx:.2f}" y1="{y1:.2f}" '
            f'x2="{cx:.2f}" y2="{y2:.2f}" '
            f'stroke="black" stroke-width="{w:.2f}" fill="none" '
            f'stroke-linecap="butt"/>'
            f'<circle cx="{cx:.2f}" cy="{y2:.2f}" '
            f'r="{w/2:.2f}" fill="black"/>'
            f'</g>'
        )
    if cap_bottom == "round":
        return (
            f'<line x1="{cx:.2f}" y1="{y1:.2f}" '
            f'x2="{cx:.2f}" y2="{y2:.2f}" '
            f'stroke="black" stroke-width="{w:.2f}" fill="none" '
            f'stroke-linecap="round"/>'
        )
    # Top round, bottom butt: butt-cap line + circle at the top.
    # Wrap in <g> for the same XML-root reason as above.
    return (
        f'<g>'
        f'<line x1="{cx:.2f}" y1="{y1:.2f}" '
        f'x2="{cx:.2f}" y2="{y2:.2f}" '
        f'stroke="black" stroke-width="{w:.2f}" fill="none" '
        f'stroke-linecap="butt"/>'
        f'<circle cx="{cx:.2f}" cy="{y1:.2f}" '
        f'r="{w/2:.2f}" fill="black"/>'
        f'</g>'
    )


def _cluster_teeth(below_pts: np.ndarray, x_gap: float = 15.0,
                   max_tooth_width: float = 40.0) -> list[np.ndarray]:
    """Cluster points by x-coordinate into tooth groups. Lenient — returns
    ALL clusters of ≥2 points. Caller is responsible for filtering by tooth
    width, point count, and shape (so it can record rejections for debug).

    Two-pass: (1) coarse cluster with `x_gap` (~15px) to bridge sparse
    tooth samples; (2) for any cluster wider than `max_tooth_width`, re-split
    it using a 1D histogram of x to find density valleys between the
    sub-tooth peaks. Without the re-split, two adjacent ticks linked by
    contour points along the inner ring stay merged into one over-wide
    cluster and get dropped."""
    if len(below_pts) < 2:
        return []

    def _coarse(pts: np.ndarray, gap: float) -> list[np.ndarray]:
        order = np.argsort(pts[:, 0])
        sorted_pts = pts[order]
        groups: list[np.ndarray] = []
        cur: list[np.ndarray] = [sorted_pts[0]]
        for i in range(1, len(sorted_pts)):
            if sorted_pts[i, 0] - sorted_pts[i - 1, 0] > gap:
                if len(cur) >= 2:
                    groups.append(np.stack(cur))
                cur = [sorted_pts[i]]
            else:
                cur.append(sorted_pts[i])
        if len(cur) >= 2:
            groups.append(np.stack(cur))
        return groups

    def _split_by_valleys(cluster: np.ndarray) -> list[np.ndarray]:
        """Re-split an over-wide cluster at x-density valleys."""
        x = cluster[:, 0]
        x_min, x_max = float(x.min()), float(x.max())
        bins = max(int(round(x_max - x_min)) + 1, 2)
        hist, edges = np.histogram(x, bins=bins, range=(x_min, x_max))
        # Valley = run of ≥5 consecutive empty bins. Looser thresholds
        # (e.g. 3) split single real ticks that happen to have a sparse
        # middle in their contour sampling; 5+ px of true blank space is
        # only seen between distinct ticks.
        in_valley = hist == 0
        valley_starts: list[int] = []
        run = 0
        for i, e in enumerate(in_valley):
            if e:
                run += 1
            else:
                if run >= 5:
                    # Cut at center of the gap.
                    valley_starts.append(i - run // 2)
                run = 0
        if not valley_starts:
            return [cluster]
        # Split at the valley positions (x values).
        cut_xs = [edges[i] for i in valley_starts]
        cut_xs.sort()
        parts: list[np.ndarray] = []
        lo = x_min - 1
        for cx in cut_xs + [x_max + 1]:
            mask = (x > lo) & (x <= cx)
            if mask.sum() >= 2:
                parts.append(cluster[mask])
            lo = cx
        return parts

    coarse = _coarse(below_pts, x_gap)
    teeth: list[np.ndarray] = []
    # Only re-split clusters in the "two-or-three ticks merged" width range.
    # Anything wider (130+ px) is almost certainly a tree whose branches
    # would be incorrectly split into phantom sub-teeth.
    SPLIT_MAX_WIDTH = max_tooth_width * 2.5
    for c in coarse:
        w = float(c[:, 0].max() - c[:, 0].min())
        if max_tooth_width < w <= SPLIT_MAX_WIDTH:
            teeth.extend(_split_by_valleys(c))
        else:
            teeth.append(c)
    return teeth


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
        min_aspect_ratio: float = 1.0,
        min_height_px: float = 6.0,
        max_width_px: float = 40.0,
        max_eigval_ratio: float = 0.7,
        tip_trim_ratio: float = 0.2,
        inner_circle: tuple[float, float, float] | None = None,
        ring_join_px: float = 15.0,
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
            fx_min, fy_min = float(full_pts[:, 0].min()), float(full_pts[:, 1].min())
            fx_max, fy_max = float(full_pts[:, 0].max()), float(full_pts[:, 1].max())
            _record(bbox=(fx_min, fy_min, fx_max, fy_max),
                    decision="reject_above", reason="entirely above horizon")
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

        # COMB-FIRST decomposition: if the (below-horizon-subset) bbox is
        # wider than a single tick, try to split it into individual teeth
        # before applying single-path shape/angle filters. vtracer often
        # merges the mountain base with all its waterfall ticks into one
        # comb-shaped contour whose bbox h/w is way below `min_aspect_ratio`
        # — without this branch every tick attached to the horizon would
        # be lost to `reject_shape`.
        if w > max_width_px:
            # Points strictly below the horizon (excluding the horizon line
            # itself). Keeps tooth corner points that sit ~1px below.
            strict_below = full_pts[full_pts[:, 1] > horizon_y]
            teeth = _cluster_teeth(strict_below, max_tooth_width=max_width_px)
            kept_teeth: list[np.ndarray] = []
            MIN_COMB_POINTS = 6
            MIN_COMB_WIDTH = 5.0
            # First pass: collect undersampled-but-tall stragglers so we can
            # merge nearby ones (the rightmost waterfall tick is often
            # traced as 3 separate near-zero-width clusters at x=1104/1115/
            # 1125 — each fails the npts/width gates alone but together
            # they describe a real tall stroke at x≈1115).
            stragglers: list[np.ndarray] = []
            STRAGGLER_MIN_HEIGHT = 100.0  # px — only merge truly-tall ones
            STRAGGLER_X_MERGE = 25.0      # px — typical tick width
            for t in teeth:
                tw = float(t[:, 0].max() - t[:, 0].min())
                th = float(t[:, 1].max() - t[:, 1].min())
                if (tw < MIN_COMB_WIDTH and th >= STRAGGLER_MIN_HEIGHT
                        and len(t) >= 2):
                    stragglers.append(t)
            stragglers.sort(key=lambda a: a[:, 0].mean())
            merged_stragglers: list[np.ndarray] = []
            i = 0
            while i < len(stragglers):
                group = [stragglers[i]]
                cx_i = stragglers[i][:, 0].mean()
                j = i + 1
                while j < len(stragglers) and stragglers[j][:, 0].mean() - cx_i <= STRAGGLER_X_MERGE:
                    group.append(stragglers[j])
                    j += 1
                if len(group) >= 2:
                    merged_stragglers.append(np.concatenate(group, axis=0))
                i = j
            teeth = teeth + merged_stragglers

            for t in teeth:
                tx_min, ty_min = float(t[:, 0].min()), float(t[:, 1].min())
                tx_max, ty_max = float(t[:, 0].max()), float(t[:, 1].max())
                tw = tx_max - tx_min
                th = ty_max - ty_min
                t_bbox = (tx_min, ty_min, tx_max, ty_max)
                if tw > max_width_px:
                    _record(bbox=t_bbox, decision="comb_tooth_reject_wide",
                            reason=f"tooth_w={tw:.0f} > {max_width_px}")
                    continue
                if tw < MIN_COMB_WIDTH or len(t) < MIN_COMB_POINTS:
                    _record(bbox=t_bbox, decision="comb_tooth_reject_undersampled",
                            reason=f"w={tw:.1f} npts={len(t)}")
                    continue
                if th < min_height_px or tw <= 0 or th / tw < min_aspect_ratio:
                    _record(bbox=t_bbox, decision="comb_tooth_reject_shape",
                            reason=f"h/w={th/(tw or 1):.1f} or h<{min_height_px}")
                    continue
                if inner_circle is not None:
                    icx, icy, ir = inner_circle
                    tcx = (tx_min + tx_max) / 2.0
                    tcy = (ty_min + ty_max) / 2.0
                    if float(np.hypot(tcx - icx, tcy - icy)) > ir - tw / 2.0:
                        _record(bbox=t_bbox, decision="comb_tooth_reject_outside",
                                reason="tooth outside inner ring")
                        continue
                kept_teeth.append(t)
            if len(kept_teeth) >= 2:
                # Keep the original path so trees/mountains/ridges survive,
                # but knock out the tick interiors with a white rectangle
                # (below horizon only) before drawing the clean snap line
                # on top. Without the knockout, the original comb's filled
                # tick interior shows through as a bulbous shape around
                # the snap line.
                out.append(path_str)
                for t in kept_teeth:
                    tx_min, ty_min = float(t[:, 0].min()), float(t[:, 1].min())
                    tx_max, ty_max = float(t[:, 0].max()), float(t[:, 1].max())
                    # Knockout: white rect covering the curvy tick interior
                    # so it doesn't show through behind the straight snap
                    # line. Start a few px BELOW horizon so we don't cut
                    # into the mountain-base stroke (which sits at the
                    # horizon line). This leaves a small visible bulge in
                    # the top ~12px of the tick where the original outline
                    # was wider than the snap line — most visible on the
                    # leftmost/rightmost ticks but minor for the others.
                    tcx = (tx_min + tx_max) / 2.0
                    knock_w = max(tx_max - tx_min, 21.0) + 6.0
                    # horizon_y sits just below the base stroke bottom
                    # edge (detected by Canny+Hough on the bottom side of
                    # the base). Starting the knockout right at horizon
                    # erases the bulge in the tick's first few px without
                    # cutting into the base. No inset needed.
                    KNOCK_TOP_INSET = 0.0
                    KNOCK_BOTTOM_PAD = 6.0
                    ky = horizon_y + KNOCK_TOP_INSET
                    kh = (ty_max - horizon_y) - KNOCK_TOP_INSET + KNOCK_BOTTOM_PAD
                    # Cap the knockout bottom at the inner ring's centerline
                    # so we don't punch a white hole through the ring stroke
                    # (otherwise the ring's lower-half stroke gets erased
                    # over the snap line's column and the tick appears to
                    # protrude past the ring).
                    if inner_circle is not None:
                        icx, icy, ir = inner_circle
                        dx_c = tcx - icx
                        if abs(dx_c) < ir:
                            ring_y_at_x = icy + float(np.sqrt(ir * ir - dx_c * dx_c))
                            max_kh = ring_y_at_x - ky
                            if max_kh > 0:
                                kh = min(kh, max_kh)
                    if kh > 0:
                        out.append(
                            f'<rect x="{tcx - knock_w/2:.2f}" '
                            f'y="{ky:.2f}" '
                            f'width="{knock_w:.2f}" '
                            f'height="{kh:.2f}" '
                            f'fill="white" stroke="none"/>'
                        )
                    out.append(("__SNAP__", tx_min, tx_max, horizon_y, ty_max,
                                True))  # top_flat for comb teeth
                    _record(bbox=(tx_min, horizon_y, tx_max, ty_max),
                            decision="snapped_comb_tooth",
                            reason=f"tooth of w={w:.0f} comb")
                continue
            # No comb found — fall through to the wide/shape rejections.
            out.append(path_str)
            _record(bbox=bbox, decision="reject_wide",
                    reason=f"w={w:.0f} > {max_width_px}")
            continue
        if h < min_height_px or w <= 0 or h / w < min_aspect_ratio:
            out.append(path_str)
            _record(bbox=bbox, decision="reject_shape",
                    reason=f"h/w={h/(w or 1):.1f} too low or h<{min_height_px}")
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
        # Short stubby paths (h <= 2w) have unreliable PCA angle/ratio
        # estimates — a few points spread over a near-square bbox can
        # easily yield 30°+ angle measurements even when the human-eye
        # intent is "vertical drip". For these we trust the bbox shape
        # (already narrow + tall enough to pass min_aspect_ratio) and
        # skip the PCA gates entirely, snapping straight to vertical.
        is_short_stubby = h <= 2.0 * w
        if not is_short_stubby:
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
        # Inner-ring gate: drop paths whose center falls outside the inner
        # ring. vtracer often emits stray fragments along the inner ring's
        # own stroke (in the annulus between the inner and outer rings).
        # Those aren't waterfall ticks — they're ring-outline noise — and
        # snapping them creates phantom vertical bars where none exist.
        cx_bbox = (x_min + x_max) / 2.0
        cy_bbox = (y_min + y_max) / 2.0
        if inner_circle is not None:
            icx, icy, ir = inner_circle
            # Distance from path center to badge center. We require the
            # path to be at least its own half-width inside the inner ring
            # so a tick that just kisses the ring is still considered "in".
            d_center = float(np.hypot(cx_bbox - icx, cy_bbox - icy))
            if d_center > ir - w / 2.0:
                out.append(path_str)
                _record(bbox=bbox, angle_deg=angle_from_vertical_deg,
                        eigval_ratio=eigval_ratio, decision="reject_outside_inner",
                        reason=f"d={d_center:.0f} > ir-w/2={ir - w/2:.0f}")
                continue

        out.append(("__SNAP__", x_min, x_max, y_min, y_max, False))
        _record(bbox=bbox, angle_deg=angle_from_vertical_deg,
                eigval_ratio=eigval_ratio, decision="snapped",
                reason="ok")

    # Resolve __SNAP__ placeholders with a UNIFORM stroke width: take the
    # median of all per-tick bbox widths so a few mis-measured edge ticks
    # don't render at the wrong thickness. The cap_inset compensation in
    # _emit_snapped_line uses this same width, so y endpoints stay aligned
    # with the original tip centres.
    widths = [t[2] - t[1] for t in out if isinstance(t, tuple) and t[0] == "__SNAP__"]
    if not widths:
        return out
    median_w = float(np.median(widths))
    resolved: list[str] = []
    for item in out:
        if isinstance(item, tuple) and item[0] == "__SNAP__":
            _, sx_min, sx_max, sy_min, sy_max, top_flat = item
            # Re-centre around the bbox-center x but force width to median_w.
            cx = (sx_min + sx_max) / 2.0
            sx_min2 = cx - median_w / 2.0
            sx_max2 = cx + median_w / 2.0
            resolved.append(_emit_snapped_line(
                sx_min2, sx_max2, sy_min, sy_max,
                inner_circle=inner_circle,
                ring_join_px=ring_join_px,
                top_flat=top_flat,
            ))
        else:
            resolved.append(item)
    return resolved


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
        "snapped":                       (0, 200,   0),   # green
        "snapped_comb_tooth":            (255, 0, 255),   # magenta
        "reject_ratio":                  (0, 165, 255),   # orange
        "reject_angle":                  (0,   0, 255),   # red
        "reject_outside_inner":          (255, 100, 0),   # cyan-orange
        "reject_above":                  (150, 150, 150), # gray
        "reject_shape":                  (180, 180, 180),
        "reject_wide":                   (200, 200, 200),
        "reject_degenerate":             (220, 220, 220),
        "skip":                          (240, 240, 240),
        # Comb-tooth-specific rejections (yellow family, distinct from
        # standalone-path rejections above).
        "comb_tooth_reject_wide":        (0, 200, 200),   # teal
        "comb_tooth_reject_undersampled":(0, 220, 220),   # pale teal
        "comb_tooth_reject_shape":       (0, 180, 180),   # darker teal
        "comb_tooth_reject_outside":     (60, 140, 200),  # blue-orange
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
