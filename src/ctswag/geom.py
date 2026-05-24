"""Extract geometric anchors from the input raster (e.g. badge center).

For text-on-arc placement, knowing the badge's center matters: both
"CAMP TINKER" and "2025" sit on arcs that are concentric with the
double ring. Finding that center first lets the optimizer search
along radii rather than over arbitrary (center_x, center_y, radius)
combinations.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import Baseline


def find_badge_center(img_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, r) of the dominant ring in the image, or None.

    Uses cv2.HoughCircles on the grayscale, with the search range scaled to
    the image size. Picks the largest detected circle (a badge's outer
    ring is typically the strongest signal).
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    H, W = gray.shape[:2]
    short = min(H, W)
    # Blur a touch to suppress edge noise
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=int(short * 0.4),
        param1=120, param2=60,
        minRadius=int(short * 0.30),
        maxRadius=int(short * 0.45),
    )
    if circles is None:
        return None
    found = circles[0]
    # Prefer the circle with the largest radius (the outer ring of a badge
    # is usually the strongest and largest signal).
    largest = max(found, key=lambda c: c[2])
    return float(largest[0]), float(largest[1]), float(largest[2])


def measure_text_bands(
        img_bgr: np.ndarray,
        cx: float, cy: float, outer_r: float, outer_sw: float,
        *,
        ink_threshold: int = 128,
        r_search_max_px: int = 300,
        n_angles: int = 720,
        r_step: float = 0.5,
        min_band_arc_deg: float = 10.0,
        angle_gap_tolerance_deg: float = 15.0,
        ) -> list[dict]:
    """Find text bands outside the outer ring by radial ink scanning.

    For each of `n_angles` angles around the badge center, walks outward
    from just past the outer ring's outer edge to `r_search_max_px` beyond,
    recording the first and last ink pixel encountered along that ray. Then
    groups contiguous angular runs (with up to `angle_gap_tolerance_deg`
    bridging) into bands and returns each band's bounds.

    Returns a list of dicts, one per band:
      {theta_min, theta_max, r_inner, r_outer, arc_deg}

    For a Camp-Tinker-style logo this typically yields two bands: one near
    theta = -pi/2 (the title arc above) and one near theta = +pi/2 (the year
    below). Anything else outside the outer ring (logos with side captions,
    chartwork in margins) would also show up as additional bands.
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    H, W = gray.shape[:2]
    r_start = outer_r + outer_sw / 2.0 + 2.0
    r_end = r_start + r_search_max_px

    angles = np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False)
    radii = np.arange(r_start, r_end, r_step)

    # Per-angle ink extent (None if no ink encountered).
    inks: list[tuple[float, float, float] | None] = [None] * n_angles
    for i, theta in enumerate(angles):
        cos_t, sin_t = math.cos(float(theta)), math.sin(float(theta))
        r_min_ink: float | None = None
        r_max_ink: float | None = None
        for r in radii:
            px = int(round(cx + r * cos_t))
            py = int(round(cy + r * sin_t))
            if not (0 <= px < W and 0 <= py < H):
                continue
            if gray[py, px] < ink_threshold:
                if r_min_ink is None:
                    r_min_ink = float(r)
                r_max_ink = float(r)
        if r_min_ink is not None and r_max_ink is not None:
            inks[i] = (float(theta), r_min_ink, r_max_ink)

    has_ink = [x is not None for x in inks]
    gap_tol = max(1, int(round(angle_gap_tolerance_deg / 360.0 * n_angles)))
    bands: list[dict] = []
    i = 0
    while i < n_angles:
        if not has_ink[i]:
            i += 1
            continue
        # Walk a contiguous run, bridging gaps up to gap_tol angles.
        start = i
        last = i
        j = i + 1
        gap_count = 0
        while j < n_angles:
            if has_ink[j]:
                last = j
                gap_count = 0
            else:
                gap_count += 1
                if gap_count > gap_tol:
                    break
            j += 1
        arc_len_deg = (last - start + 1) / n_angles * 360.0
        if arc_len_deg < min_band_arc_deg:
            i = j
            continue
        r_inners = [inks[k][1] for k in range(start, last + 1) if inks[k] is not None]
        r_outers = [inks[k][2] for k in range(start, last + 1) if inks[k] is not None]
        bands.append({
            "theta_min": float(angles[start]),
            "theta_max": float(angles[last]),
            "r_inner": float(min(r_inners)),
            "r_outer": float(max(r_outers)),
            "arc_deg": float(arc_len_deg),
            "cx": float(cx),
            "cy": float(cy),
        })
        i = j

    return bands


def band_to_baseline(band: dict) -> Baseline:
    """Construct an arc baseline directly from a measured text band.

    The baseline radius is the EDGE of the band closer to the badge ring
    (not the centerline). In SVG `textPath`, glyphs sit ON the path with
    their baselines and grow perpendicular to the path direction:

      Top arcs  (sin(t_mid) < 0): text grows AWAY from badge center, so
                                  the glyph BASELINE is at the inner edge
                                  of the band (closer to badge).
      Bottom arcs (sin(t_mid) > 0): text grows TOWARD badge center, so the
                                    baseline is at the outer edge of the
                                    band (farther from badge).

    Using the centerline would put the text in the inner half of the
    measured band -- visibly too close to the ring for bottom text and
    too far from the ring for top text.

    The (t0, t1) pair is also ordered so the leftmost endpoint is first
    (same rule as in band_to_polygon), so SVG textPath lays out characters
    in reading order without the path being traversed the wrong way.
    """
    cx, cy = band["cx"], band["cy"]
    t_min, t_max = band["theta_min"], band["theta_max"]
    t_mid = (t_min + t_max) / 2.0
    if math.sin(t_mid) < 0:
        # Top arc -- baseline at the side closer to the badge.
        r_baseline = band["r_inner"]
    else:
        # Bottom arc -- baseline at the side closer to the badge.
        r_baseline = band["r_outer"]
    # Reading order: leftmost endpoint first.
    if math.cos(t_min) > math.cos(t_max):
        t0, t1 = t_max, t_min
    else:
        t0, t1 = t_min, t_max
    return Baseline(
        kind="arc",
        params=(float(cx), float(cy), float(r_baseline), float(t0), float(t1)),
        residual=0.0,
    )


def band_to_polygon(band: dict, n: int = 24) -> np.ndarray:
    """Build an arc-shaped polygon (N+1 outer points + N+1 inner points) that
    follows a measured text band's angular and radial extent. Suitable as a
    direct replacement for the title/year polygons synthesized by
    detect_camp_geom.

    The vertices are ordered so the FIRST polygon point is at the band's
    leftmost x. baseline.fit reads vertex order to assign (t0, t1) on the
    arc, and the SVG textPath then lays out characters from t0 to t1; if we
    started at the right, "2025" would render as "5202".

    For top arcs (180° < theta < 360°) increasing theta moves left->right,
    so the natural order is fine. For bottom arcs (0° < theta < 180°)
    increasing theta moves right->left, so we reverse.
    """
    cx, cy = band["cx"], band["cy"]
    t0, t1 = band["theta_min"], band["theta_max"]
    r_i, r_o = band["r_inner"], band["r_outer"]
    if math.cos(t0) > math.cos(t1):
        # The endpoint at t0 has larger x than at t1, so t0 is on the right.
        # Walk from t1 (left) to t0 (right) instead.
        t0, t1 = t1, t0
    upper: list[tuple[float, float]] = []
    lower: list[tuple[float, float]] = []
    for k in range(n + 1):
        t = t0 + (t1 - t0) * k / n
        cos_t, sin_t = math.cos(t), math.sin(t)
        upper.append((cx + r_o * cos_t, cy + r_o * sin_t))
        lower.append((cx + r_i * cos_t, cy + r_i * sin_t))
    return np.array(upper + lower[::-1], dtype=np.float32)
