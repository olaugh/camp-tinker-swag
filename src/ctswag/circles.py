"""Detect ideal concentric rings in the input raster.

Strategy (see eval/circle_audit.py for the diagnostic that proved it out):
  1. Run HoughCircles permissively to get one strong anchor (the badge's
     dominant ring -- the strongest Hough hit is rock-solid even when
     centers wobble for the weaker concentric rings).
  2. Lock that center, then sweep radius from r_lo to r_hi measuring
     "ink at exactly this radius" along 720 perimeter angles (1-pixel
     band -- NOT the full stroke). The resulting coverage(r) curve has
     a clear plateau of 1.0 over each ring's stroke width with sharp
     transitions at the inner/outer edges.
  3. Each connected run of high coverage is one ring; report its
     midpoint radius as the centerline and its width as the stroke.

This handles the case where Hough's per-circle center fits drift by
50-100 px for weaker rings (sun rays / mountains attached to the inner
ring make the gradient response asymmetric). The anchor's center stays
trustworthy because the strongest ring is fully closed and unobstructed.
"""
from __future__ import annotations

import cv2
import numpy as np


def _hough_anchor(gray: np.ndarray,
                  *, min_radius_ratio: float, max_radius_ratio: float
                  ) -> tuple[float, float, float] | None:
    """Return the single strongest Hough circle as (cx, cy, r), or None."""
    H, W = gray.shape[:2]
    short = min(H, W)
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=max(int(short * 0.02), 8),
        param1=120, param2=50,
        minRadius=int(short * min_radius_ratio),
        maxRadius=int(short * max_radius_ratio),
    )
    if circles is None or len(circles[0]) == 0:
        return None
    c = circles[0][0]  # HoughCircles returns strongest first
    return float(c[0]), float(c[1]), float(c[2])


def _perimeter_coverage(gray: np.ndarray, cx: float, cy: float, r: float,
                        *, n_samples: int = 720,
                        ink_threshold: int = 128) -> float:
    """Fraction of n_samples angles where the pixel at exactly (cx + r*cos t,
    cy + r*sin t) is ink (gray < threshold). Angles whose sample falls
    out-of-bounds are excluded from the denominator."""
    H, W = gray.shape[:2]
    hits = 0
    counted = 0
    for i in range(n_samples):
        theta = 2 * np.pi * i / n_samples
        px = int(round(cx + r * np.cos(theta)))
        py = int(round(cy + r * np.sin(theta)))
        if not (0 <= px < W and 0 <= py < H):
            continue
        counted += 1
        if gray[py, px] < ink_threshold:
            hits += 1
    return hits / max(counted, 1)


def detect(img_bgr: np.ndarray,
           *,
           min_radius_ratio: float = 0.10,
           max_radius_ratio: float = 0.48,
           anchor_min_ratio: float = 0.20,
           anchor_max_ratio: float = 0.48,
           coverage_threshold: float = 0.85,
           min_stroke_px: int = 3,
           max_rings: int = 6) -> list[tuple[float, float, float]]:
    """Return concentric rings as (cx, cy, r), biggest first.

    All returned rings share the same (cx, cy) anchored on the strongest
    Hough hit. Each entry's `r` is the centerline of the ring's stroke
    (use `estimate_stroke_width` to recover the stroke thickness, or call
    `detect_with_strokes` which returns both).
    """
    rings = detect_with_strokes(
        img_bgr,
        min_radius_ratio=min_radius_ratio,
        max_radius_ratio=max_radius_ratio,
        anchor_min_ratio=anchor_min_ratio,
        anchor_max_ratio=anchor_max_ratio,
        coverage_threshold=coverage_threshold,
        min_stroke_px=min_stroke_px,
        max_rings=max_rings,
    )
    return [(cx, cy, r) for (cx, cy, r, _w) in rings]


def _refine_ring(gray: np.ndarray,
                 cx: float, cy: float, r: float, stroke: float,
                 *,
                 dxy: float = 2.0, dr: float = 4.0, ds: float = 2.0,
                 xy_step: float = 0.5, r_step: float = 0.5, s_step: float = 1.0,
                 s_step_fine: float = 0.05, s_fine_radius: float = 1.5,
                 lock_center: tuple[float, float] | None = None
                 ) -> tuple[float, float, float, float, float]:
    """Local sweep around (cx, cy, r, stroke) minimising rendered-vs-source L1
    inside a tight annulus, then a stroke-only refinement at sub-pixel step.

    Rendering is anti-aliased: each pixel's intensity is the coverage of the
    ring's stroke over that pixel, modelled as a linear ramp across the
    boundary (a pixel whose center is exactly on the stroke edge contributes
    0.5 ink). Without anti-aliasing the L1 is piecewise constant in stroke
    and sub-pixel stroke sweeps do nothing.

    Two-stage:
      1. Coarse: sweep (cx, cy, r, stroke) at the supplied step sizes.
      2. Fine: lock (cx, cy, r) at stage-1 best; sweep stroke alone over
         +/- `s_fine_radius` at `s_step_fine` resolution.

    `lock_center` pins (cx, cy) -- used for inner rings, which must be
    concentric with the outer ring's already-refined center.

    Returns (cx_best, cy_best, r_best, stroke_best, l1_best).
    """
    H, W = gray.shape[:2]
    # Wide search annulus around the seed ring -- large enough to contain
    # every candidate's stroke.
    margin = int(stroke) + int(dr) + int(ds) + 6
    yy, xx = np.mgrid[0:H, 0:W]
    dist_seed = np.hypot(xx - cx, yy - cy)
    search_mask = np.abs(dist_seed - r) <= margin
    if not search_mask.any():
        return cx, cy, r, stroke, float("inf")

    src_ann = gray[search_mask].astype(np.float32)
    xs_ann = xx[search_mask].astype(np.float32)
    ys_ann = yy[search_mask].astype(np.float32)

    def _ring_l1(d_minus_r: np.ndarray, s: float) -> float:
        # Anti-aliased coverage:
        #   |d - r| <  s/2 - 0.5  -> fully ink   (rendered = 0)
        #   |d - r| >  s/2 + 0.5  -> fully paper (rendered = 255)
        #   else linear ramp.
        # Continuous in s, so sub-pixel sweeps actually change the score.
        rendered = 255.0 * np.clip(d_minus_r - s / 2.0 + 0.5, 0.0, 1.0)
        return float(np.abs(rendered - src_ann).mean())

    if lock_center is not None:
        xy_iter = [(lock_center[0] - cx, lock_center[1] - cy)]
    else:
        xys = np.arange(-dxy, dxy + xy_step / 2, xy_step)
        xy_iter = [(float(dx), float(dy)) for dx in xys for dy in xys]
    rs = np.arange(-dr, dr + r_step / 2, r_step)
    ss = np.arange(-ds, ds + s_step / 2, s_step)

    best = (cx, cy, r, stroke, float("inf"))
    # Stage 1: coarse joint sweep.
    for dxc, dyc in xy_iter:
        cx_c = cx + dxc
        cy_c = cy + dyc
        d = np.hypot(xs_ann - cx_c, ys_ann - cy_c)
        for drc in rs:
            r_c = r + float(drc)
            if r_c <= 0:
                continue
            d_minus_r = np.abs(d - r_c)
            for dsc in ss:
                s_c = stroke + float(dsc)
                if s_c <= 0:
                    continue
                l1 = _ring_l1(d_minus_r, s_c)
                if l1 < best[4]:
                    best = (cx_c, cy_c, r_c, s_c, l1)

    # Stage 2: lock (cx, cy, r); refine stroke at sub-pixel step.
    cx_b, cy_b, r_b, s_b, _ = best
    d = np.hypot(xs_ann - cx_b, ys_ann - cy_b)
    d_minus_r = np.abs(d - r_b)
    ss_fine = np.arange(s_b - s_fine_radius,
                        s_b + s_fine_radius + s_step_fine / 2,
                        s_step_fine)
    for s_c in ss_fine:
        s_c = float(s_c)
        if s_c <= 0:
            continue
        l1 = _ring_l1(d_minus_r, s_c)
        if l1 < best[4]:
            best = (cx_b, cy_b, r_b, s_c, l1)

    return best


def detect_with_strokes(
        img_bgr: np.ndarray,
        *,
        min_radius_ratio: float = 0.10,
        max_radius_ratio: float = 0.48,
        anchor_min_ratio: float = 0.20,
        anchor_max_ratio: float = 0.48,
        coverage_threshold: float = 0.85,
        min_stroke_px: int = 3,
        max_rings: int = 6,
        refine: bool = True) -> list[tuple[float, float, float, float]]:
    """Same as detect() but returns (cx, cy, r, stroke_px) per ring.

    When `refine=True`, each ring is locally optimised by `_refine_ring`
    against the source gray. The outer (biggest) ring is fit freely; inner
    rings re-use the outer's refined center to enforce concentricity.
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    H, W = gray.shape[:2]
    short = min(H, W)

    anchor = _hough_anchor(gray, min_radius_ratio=anchor_min_ratio,
                            max_radius_ratio=anchor_max_ratio)
    if anchor is None:
        return []
    acx, acy, _ = anchor

    r_lo = int(short * min_radius_ratio)
    r_hi = int(short * max_radius_ratio)
    radii = np.arange(r_lo, r_hi + 1, 1.0)
    coverage = np.array([
        _perimeter_coverage(gray, acx, acy, float(r)) for r in radii
    ])

    # Walk connected runs of coverage >= threshold. Each run is one ring.
    above = coverage >= coverage_threshold
    rings: list[tuple[float, float, float, float]] = []
    i = 0
    while i < len(above):
        if not above[i]:
            i += 1
            continue
        j = i
        while j < len(above) and above[j]:
            j += 1
        if j - i >= min_stroke_px:
            mid_idx = (i + j - 1) // 2
            stroke = float(radii[j - 1] - radii[i] + 1)
            rings.append((acx, acy, float(radii[mid_idx]), stroke))
        i = j

    rings.sort(key=lambda r: -r[2])
    rings = rings[:max_rings]

    if not refine or not rings:
        return rings

    # Refine the outer ring freely, then lock its center for the inner rings.
    refined: list[tuple[float, float, float, float]] = []
    cx_o, cy_o, r_o, s_o = rings[0]
    cx_r, cy_r, r_r, s_r, _l1 = _refine_ring(gray, cx_o, cy_o, r_o, s_o)
    refined.append((cx_r, cy_r, r_r, s_r))
    for (cx_i, cy_i, r_i, s_i) in rings[1:]:
        cxr, cyr, rr, sr, _ = _refine_ring(gray, cx_i, cy_i, r_i, s_i,
                                            lock_center=(cx_r, cy_r))
        refined.append((cxr, cyr, rr, sr))
    return refined


def estimate_stroke_width(img_bgr: np.ndarray, cx: float, cy: float, r: float,
                          *, samples: int = 24, max_probe: int = 30) -> float:
    """Sample the ring's stroke width by walking outward from each sample
    point along the radial direction until ink ends.

    Returns the median measured thickness in pixels. Falls back to 6 if no
    samples land on ink. Kept for callers that have a ring but not the
    stroke from `detect_with_strokes`.
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    H, W = gray.shape[:2]
    widths: list[int] = []
    for i in range(samples):
        theta = 2 * np.pi * i / samples
        dx, dy = np.cos(theta), np.sin(theta)
        rx = int(round(cx + r * dx))
        ry = int(round(cy + r * dy))
        if not (0 <= rx < W and 0 <= ry < H) or gray[ry, rx] > 128:
            continue
        inner = 0
        for k in range(1, max_probe):
            qx = int(round(cx + (r - k) * dx))
            qy = int(round(cy + (r - k) * dy))
            if not (0 <= qx < W and 0 <= qy < H) or gray[qy, qx] > 128:
                inner = k
                break
        outer = 0
        for k in range(1, max_probe):
            qx = int(round(cx + (r + k) * dx))
            qy = int(round(cy + (r + k) * dy))
            if not (0 <= qx < W and 0 <= qy < H) or gray[qy, qx] > 128:
                outer = k
                break
        widths.append(inner + outer)
    if not widths:
        return 6.0
    return float(np.median(widths))


def paint_white(image_bgr: np.ndarray,
                circles: list[tuple[float, float, float]],
                widths: list[float],
                *, pad_px: int = -1) -> np.ndarray:
    """Paint each detected ring white in-place so a downstream tracer doesn't
    re-vectorize it. Returns the modified image.

    `pad_px` widens the painted band by this many pixels on EACH side beyond
    the stroke. Default -1: paint a band 2 px narrower than the stroke.

    Why a negative default: in the source raster, line-art strokes (sun
    rays, mountain ridges, the baseline) continue 1-2 px *into* the ring's
    stroke before terminating -- visually they "end at the ring" but the
    ink overlaps. Painting exactly the stroke (or wider) wipes out those
    overlap pixels and the rendered output ends up with a thin gap. Leaving
    1 px of original ring ink on each side preserves the line-art overlap;
    the leftover ring ink is harmless because the SVG `<circle>` (drawn on
    top of the traced geometry in assemble.py, with a slightly oversized
    stroke) covers it.
    """
    out = image_bgr.copy()
    for (cx, cy, r), w in zip(circles, widths):
        thickness = max(int(round(w)) + 2 * pad_px, 1)
        color = (255, 255, 255) if out.ndim == 3 else 255
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)),
                   color, thickness=thickness)
    return out
