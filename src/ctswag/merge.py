"""Merge nearby polygons that share an arc baseline (CAMP + TINKER -> CAMP TINKER)."""
from __future__ import annotations

import math

import numpy as np
from skimage.measure import CircleModel


def _polygon_centerline(poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(poly, dtype=np.float64)
    if len(pts) % 2 == 0 and len(pts) >= 4:
        half = len(pts) // 2
        top = pts[:half]
        bot = pts[half:][::-1]
        m = min(len(top), len(bot))
        return (top[:m] + bot[:m]) / 2
    return pts


def merge_arc(polygons: list[np.ndarray], texts: list[str],
              *, residual_threshold: float = 8.0,
              radius_min: float = 50.0, radius_max: float = 2000.0) -> list[tuple[np.ndarray, str]]:
    """Merge polygons whose centerlines all lie on a common circle.

    Returns list of (merged_polygon, joined_text). Polygons that don't fit a
    common arc are returned individually.
    """
    if len(polygons) < 2:
        return [(p, t) for p, t in zip(polygons, texts)]

    # Compute centerline per polygon
    centerlines = [_polygon_centerline(p) for p in polygons]
    all_pts = np.vstack(centerlines)
    if len(all_pts) < 3:
        return [(p, t) for p, t in zip(polygons, texts)]

    try:
        cm = CircleModel.from_estimate(all_pts)
        cx, cy = cm.center
        r = cm.radius
        dists = np.sqrt((all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2)
        residual = float(np.mean(np.abs(dists - r)))
    except Exception:
        return [(p, t) for p, t in zip(polygons, texts)]

    if (residual > residual_threshold or
            not (radius_min < r < radius_max)):
        return [(p, t) for p, t in zip(polygons, texts)]

    # Sort polygons left-to-right around the arc (by angular position of centroid)
    items = []
    for poly, text in zip(polygons, texts):
        c = poly.mean(axis=0)
        theta = math.atan2(c[1] - cy, c[0] - cx)
        items.append((theta, poly, text))
    items.sort(key=lambda x: x[0])
    merged_poly = np.vstack([p for _, p, _ in items])
    joined_text = " ".join(t for _, _, t in items if t)
    return [(merged_poly, joined_text)]


def maybe_merge(polygons: list[np.ndarray], texts: list[str],
                *, residual_threshold: float = 8.0,
                max_gap_deg: float = 45.0) -> list[tuple[np.ndarray, str]]:
    """Cluster polygons into arc-sharing groups; merge each group.

    Two polygons can only be merged if (a) all polygons in the group fit a
    common arc with small per-polygon residual, AND (b) no two adjacent
    polygons along the arc have an angular gap > `max_gap_deg` -- i.e. they
    have to read like consecutive words, not separate text blocks.

    Greedy: pick the polygon-pair with the smallest joint-arc residual,
    grow that group as long as adding a polygon keeps residual < threshold,
    repeat with the leftovers. Singletons are returned as-is.
    """
    items = [(p, t) for p, t in zip(polygons, texts)]
    if len(items) < 2:
        return items

    def arc_residual(group_polys: list[np.ndarray]) -> tuple[float, float]:
        """Returns (worst per-polygon residual, radius).

        Also enforces: no two adjacent polygons (sorted by centroid angle on
        the fitted arc) may have a gap > max_gap_deg.
        """
        try:
            all_pts = np.vstack([_polygon_centerline(p) for p in group_polys])
            cm = CircleModel.from_estimate(all_pts)
            cx, cy = cm.center; r = cm.radius
            worst = 0.0
            for poly in group_polys:
                cl = _polygon_centerline(poly)
                d = np.sqrt((cl[:, 0]-cx)**2 + (cl[:, 1]-cy)**2)
                worst = max(worst, float(np.mean(np.abs(d - r))))
            # Adjacent-gap check (gap = arc distance between adjacent polygons'
            # nearest endpoints, NOT between their centroids).
            if len(group_polys) >= 2:
                spans = []  # (theta_min, theta_max) per polygon
                for poly in group_polys:
                    theta = np.arctan2(poly[:, 1] - cy, poly[:, 0] - cx)
                    spans.append((float(theta.min()), float(theta.max())))
                spans.sort(key=lambda s: (s[0] + s[1]) / 2)  # sort by midpoint
                max_gap = 0.0
                for (a, b), (c2, d2) in zip(spans, spans[1:]):
                    gap = abs(c2 - b)  # next's min minus prev's max
                    max_gap = max(max_gap, gap)
                if math.degrees(max_gap) > max_gap_deg:
                    return float("inf"), r
            return worst, r
        except Exception:
            return float("inf"), 0.0

    remaining = list(range(len(items)))
    out: list[tuple[np.ndarray, str]] = []
    while remaining:
        # Find best-pair to start a group; if no pair fits, dump remaining as singletons.
        best_pair = None
        best_resid = float("inf")
        for i_idx, i in enumerate(remaining):
            for j in remaining[i_idx+1:]:
                resid, _ = arc_residual([items[i][0], items[j][0]])
                if resid < best_resid:
                    best_resid = resid
                    best_pair = (i, j)
        if best_pair is None or best_resid > residual_threshold:
            # No good pair; dump all remaining as singletons.
            for i in remaining:
                out.append(items[i])
            break
        group = list(best_pair)
        # Greedily extend group.
        leftover = [k for k in remaining if k not in group]
        improved = True
        while improved and leftover:
            improved = False
            best_extend = None
            best_extend_resid = float("inf")
            for k in leftover:
                resid, _ = arc_residual([items[g][0] for g in group] + [items[k][0]])
                if resid < best_extend_resid:
                    best_extend_resid = resid
                    best_extend = k
            if best_extend is not None and best_extend_resid <= residual_threshold:
                group.append(best_extend)
                leftover.remove(best_extend)
                improved = True
        # Sort group left-to-right along the arc.
        polys = [items[g][0] for g in group]
        texts_g = [items[g][1] for g in group]
        all_pts = np.vstack([_polygon_centerline(p) for p in polys])
        cm = CircleModel.from_estimate(all_pts)
        cx, cy = cm.center
        angles = [math.atan2(p.mean(axis=0)[1] - cy, p.mean(axis=0)[0] - cx) for p in polys]
        order = np.argsort(angles)
        merged_poly = np.vstack([polys[i] for i in order])
        joined_text = " ".join(texts_g[i] for i in order if texts_g[i])
        out.append((merged_poly, joined_text))
        remaining = leftover
    return out
