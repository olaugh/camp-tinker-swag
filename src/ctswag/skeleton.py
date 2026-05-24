"""Skeletonise the inside-the-rings line art and snap endpoints to neighbors.

Assumptions baked into this module:
  * All line art inside the rings has a single fixed stroke width.
  * Lines that visually appear to touch at intersections actually meet --
    small gaps in the source raster (e.g. sun rays ending 3-5 px short of
    the inner ring) are artifacts of the source and should be closed.

These let us replace vtracer (which fits filled cubic-Bezier outlines around
each stroke) with a cleaner two-step: extract centerlines via
`skimage.morphology.skeletonize`, snap endpoints near the ring to the ring,
and re-stroke at the known width. The output has no anti-aliasing noise,
no jagged Bezier wiggle, and is editable as polylines.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage.draw import line as _bresenham
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize as _sk_skeletonize


def to_binary(layer_rgb_or_gray: np.ndarray, *, ink_threshold: int = 128) -> np.ndarray:
    """Convert an ink-on-white image to a boolean mask (True = ink)."""
    if layer_rgb_or_gray.ndim == 3:
        gray = cv2.cvtColor(layer_rgb_or_gray, cv2.COLOR_RGB2GRAY)
    else:
        gray = layer_rgb_or_gray
    return gray < ink_threshold


def skeletonize_binary(binary: np.ndarray) -> np.ndarray:
    """1-pixel-wide centerline mask. Wraps `skimage.morphology.skeletonize`."""
    return _sk_skeletonize(binary.astype(bool))


_NEIGHBOR_KERNEL = np.array([[1, 1, 1],
                              [1, 0, 1],
                              [1, 1, 1]], dtype=np.uint8)


def find_endpoints(skel: np.ndarray) -> np.ndarray:
    """Return (N, 2) array of (y, x) for every skeleton pixel that has
    exactly one 8-connected neighbor in the skeleton."""
    sk = skel.astype(np.uint8)
    n_neighbors = ndimage.convolve(sk, _NEIGHBOR_KERNEL, mode="constant", cval=0)
    is_endpoint = (sk == 1) & (n_neighbors == 1)
    return np.argwhere(is_endpoint)


def _walk_branch(skel: np.ndarray, sy: int, sx: int) -> list[tuple[int, int]]:
    """Walk along a degree-1 spur from endpoint (sy, sx) until the first
    junction (degree >= 3) or the far endpoint. Returns the list of (y, x)
    pixels in the branch, strictly BEFORE the junction.

    If the start isn't actually a degree-1 endpoint, returns [(sy, sx)] only.
    """
    H, W = skel.shape
    path: list[tuple[int, int]] = [(sy, sx)]
    prev: tuple[int, int] | None = None
    cur = (sy, sx)
    while True:
        cy, cx = cur
        # Collect ink neighbors excluding the one we came from.
        forward: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if not skel[ny, nx]:
                    continue
                if prev is not None and (ny, nx) == prev:
                    continue
                forward.append((ny, nx))
        # cur is part of the branch only while degree-along-walk == 1.
        if len(forward) != 1:
            break
        prev = cur
        cur = forward[0]
        path.append(cur)
    return path


def prune_spurs(skel: np.ndarray, *, max_spur_len: int = 8) -> np.ndarray:
    """Iteratively delete dead-end branches shorter than `max_spur_len`.

    Skeletonising a thick stroke produces 2-4 px "Y" forks at every cap
    because the medial axis of a stroke cap branches. Pruning is just
    "any degree-1 branch shorter than the stroke width is a cap artifact,
    not real structure".

    Iterates because deleting one arm of a Y may turn the junction into
    a new endpoint that needs re-evaluation.
    """
    result = skel.copy()
    while True:
        endpoints = find_endpoints(result)
        if endpoints.size == 0:
            break
        removed_any = False
        for ey, ex in endpoints:
            if not result[ey, ex]:
                continue  # already deleted in this pass
            branch = _walk_branch(result, int(ey), int(ex))
            if len(branch) <= max_spur_len:
                for py, px in branch:
                    result[py, px] = False
                removed_any = True
        if not removed_any:
            break
    return result


def _endpoint_direction(skel: np.ndarray, ey: int, ex: int,
                         *, n_back: int = 10) -> tuple[float, float]:
    """Unit vector along the branch's principal axis, pointing AWAY from
    the body of the branch (i.e. outward from the endpoint).

    Walks up to `n_back` pixels inward from the endpoint along the branch,
    stopping early at junctions (degree >= 3) or other endpoints. Fits a
    line through the collected positions via the principal eigenvector of
    their covariance and returns it oriented toward the endpoint.

    Single-pixel "neighbor direction" inherits any 1-2 px off-axis tilt
    left by spur pruning (the new endpoint sits at the old Y junction).
    Averaging over ~10 px washes that out so extensions stay parallel to
    the true ray axis instead of veering to one side.
    """
    H, W = skel.shape
    positions: list[tuple[int, int]] = [(ey, ex)]
    prev: tuple[int, int] | None = None
    cur = (ey, ex)
    for _ in range(n_back):
        cy, cx = cur
        forward: list[tuple[int, int]] = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if not (0 <= ny < H and 0 <= nx < W):
                    continue
                if not skel[ny, nx]:
                    continue
                if prev is not None and (ny, nx) == prev:
                    continue
                forward.append((ny, nx))
        if len(forward) != 1:
            # Junction or far endpoint: stop walking.
            break
        prev = cur
        cur = forward[0]
        positions.append(cur)
    if len(positions) < 2:
        return 0.0, 0.0
    pts = np.asarray(positions, dtype=np.float64)
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    # Principal axis = eigvec of largest eigval of covariance matrix.
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, -1]
    # Flip if the axis points away from the endpoint.
    to_endpoint = np.array([float(ey), float(ex)]) - centroid
    if float(axis @ to_endpoint) < 0:
        axis = -axis
    norm = float(np.linalg.norm(axis))
    if norm < 1e-6:
        return 0.0, 0.0
    axis = axis / norm
    return float(axis[0]), float(axis[1])


def _node_mask(skel: np.ndarray) -> np.ndarray:
    """Boolean mask: True at skeleton pixels that aren't degree-2 (i.e.
    endpoints with degree 1 or junctions with degree >= 3)."""
    sk = skel.astype(np.uint8)
    nb = ndimage.convolve(sk, _NEIGHBOR_KERNEL, mode="constant", cval=0)
    return (sk == 1) & (nb != 2)


def _ink_neighbors(skel: np.ndarray, y: int, x: int,
                   exclude: tuple[int, int] | None = None
                   ) -> list[tuple[int, int]]:
    H, W = skel.shape
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if not (0 <= ny < H and 0 <= nx < W):
                continue
            if not skel[ny, nx]:
                continue
            if exclude is not None and (ny, nx) == exclude:
                continue
            out.append((int(ny), int(nx)))
    return out


def _last_direction(path: list[tuple[int, int]], n_back: int = 8
                    ) -> tuple[float, float]:
    """Unit vector from `n_back` steps ago to the last pixel of path."""
    if len(path) < 2:
        return 0.0, 0.0
    k = min(n_back, len(path) - 1)
    dy = path[-1][0] - path[-1 - k][0]
    dx = path[-1][1] - path[-1 - k][1]
    norm = float(np.hypot(dy, dx))
    if norm < 1e-6:
        return 0.0, 0.0
    return dy / norm, dx / norm


def straighten_segments(skel: np.ndarray,
                         *, min_length: int = 8,
                         straightness_ratio: float = 0.05,
                         collinear_threshold_deg: float = 25.0,
                         axis_snap_deg: float = 5.0,
                         ) -> tuple[np.ndarray, int]:
    """Walk every node-to-node segment of the skeleton (nodes = endpoints
    or junctions), and replace any segment whose pixels lie on a line (PCA
    minor/major eigenvalue ratio below threshold) with an ideal Bresenham
    line through the segment's two endpoints.

    When the walk reaches a junction, we continue THROUGH it if one of the
    outgoing branches is collinear with the incoming direction (within
    `collinear_threshold_deg`). This subordinates "branch" attachments
    (snowcaps joining a mountain ridge, tree branches off a trunk) to the
    main edge: the ridge from peak to base stays one segment, the snowcap
    becomes a separate side-branch walked independently.

    `straightness_ratio` is `eigval_minor / eigval_major` of the segment's
    pixel covariance. 0 = perfect line; curves give higher values. 0.05
    accepts the stair-stepping of a diagonal Bresenham-like skeleton but
    rejects mountain ridges with a peak in the middle.

    `axis_snap_deg` snaps the PCA axis to EXACTLY horizontal or vertical
    when the fitted axis is within that many degrees. The source's mountain
    baseline is exactly horizontal; waterfall ticks and tree trunks are
    exactly vertical. PCA fits these to within ~1 deg, but skeleton
    stair-stepping leaves a subtle tilt; this snap removes it. Set to 0 to
    disable (segments stay at their PCA-fitted angle).

    Returns (new_skeleton, n_segments_straightened).
    """
    H, W = skel.shape
    is_node = _node_mask(skel)
    # Each undirected edge has TWO oriented starts: (node_a, first_step_a)
    # and (node_b, first_step_b). Mark both as visited once walked.
    visited_starts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    edges_to_fit: list[list[tuple[int, int]]] = []

    node_ys, node_xs = np.where(is_node)
    for sy, sx in zip(node_ys, node_xs):
        start = (int(sy), int(sx))
        for first in _ink_neighbors(skel, start[0], start[1]):
            if (start, first) in visited_starts:
                continue
            path: list[tuple[int, int]] = [start, first]
            prev: tuple[int, int] = start
            cur: tuple[int, int] = first
            steps = 0
            max_steps = H + W  # safety against pathological loops
            while True:
                steps += 1
                if steps > max_steps:
                    break
                # If cur is a junction, try to continue through if one of
                # the outgoing branches is collinear with the incoming.
                if is_node[cur]:
                    forward = _ink_neighbors(skel, cur[0], cur[1], exclude=prev)
                    # Endpoint (no forward neighbours) -> stop.
                    if not forward:
                        break
                    # Junction: pick most-collinear branch.
                    in_dy, in_dx = _last_direction(path, n_back=8)
                    if in_dy == 0.0 and in_dx == 0.0:
                        break
                    best_dot = -1.0
                    best_next: tuple[int, int] | None = None
                    for nb in forward:
                        out_dy = nb[0] - cur[0]
                        out_dx = nb[1] - cur[1]
                        norm = float(np.hypot(out_dy, out_dx))
                        if norm < 1e-6:
                            continue
                        dot = (in_dy * out_dy + in_dx * out_dx) / norm
                        if dot > best_dot:
                            best_dot = dot
                            best_next = nb
                    cos_thresh = float(np.cos(np.deg2rad(collinear_threshold_deg)))
                    if best_dot < cos_thresh or best_next is None:
                        break  # no collinear continuation -> stop at junction
                    # Step through the junction.
                    prev = cur
                    cur = best_next
                    path.append(cur)
                    continue
                # Mid-line pixel (degree 2): exactly one forward neighbour.
                forward = _ink_neighbors(skel, cur[0], cur[1], exclude=prev)
                if not forward:
                    break
                prev = cur
                cur = forward[0]
                path.append(cur)
                if cur == start:
                    break  # closed loop -- stop before infinite-looping
            visited_starts.add((start, first))
            if len(path) >= 2:
                # Reverse direction of the same undirected edge.
                visited_starts.add((path[-1], path[-2]))
            edges_to_fit.append(path)

    result = skel.copy()
    n_fitted = 0
    sin_snap = float(np.sin(np.deg2rad(axis_snap_deg))) if axis_snap_deg > 0 else 0.0
    for path in edges_to_fit:
        if len(path) < min_length:
            continue
        pts = np.asarray(path, dtype=np.float64)
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals[1] < 1e-6:
            continue
        if eigvals[0] / eigvals[1] > straightness_ratio:
            continue  # too curved -- not a straight stroke

        # Internal junctions where other edges attach (snowcap T on a
        # ridge, tier branch on a tree trunk). Wiping these would orphan
        # the attached edges; we draw segment-by-segment between them.
        internal_node_idx = [i for i in range(1, len(path) - 1)
                              if is_node[path[i]]]

        # Axis snap: if the PCA axis is within `axis_snap_deg` of horizontal
        # or vertical AND the path has no internal junctions, draw a single
        # exactly-axis-aligned line at the median row/column. Source design
        # has the mountain baseline exactly horizontal and the waterfall
        # ticks / tree trunk segments exactly vertical, but PCA-then-
        # Bresenham leaves a 1-2 px tilt from skeleton stair-stepping. We
        # skip paths with internal junctions because snapping them would
        # also have to move the junction pixel onto the snapped line, which
        # would tug the attached edges (next pass would handle that).
        axis = eigvecs[:, -1]
        if (sin_snap > 0 and not internal_node_idx
                and (abs(axis[0]) < sin_snap or abs(axis[1]) < sin_snap)):
            if abs(axis[0]) < sin_snap:
                # Horizontal: fix y at the path's median row.
                y_line = int(round(float(np.median(pts[:, 0]))))
                x_min = int(round(float(pts[:, 1].min())))
                x_max = int(round(float(pts[:, 1].max())))
                rr, cc = _bresenham(y_line, x_min, y_line, x_max)
            else:
                # Vertical: fix x at the path's median column.
                x_line = int(round(float(np.median(pts[:, 1]))))
                y_min = int(round(float(pts[:, 0].min())))
                y_max = int(round(float(pts[:, 0].max())))
                rr, cc = _bresenham(y_min, x_line, y_max, x_line)
            for py, px in path:
                if not is_node[py, px]:
                    result[py, px] = False
            keep = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            result[rr[keep], cc[keep]] = True
            n_fitted += 1
            continue

        # Generic case: PCA-fitted Bresenham, segment-by-segment between
        # any internal junctions so attached edges stay connected.
        breakpoints = [0] + internal_node_idx + [len(path) - 1]
        for k in range(len(breakpoints) - 1):
            i_a, i_b = breakpoints[k], breakpoints[k + 1]
            for j in range(i_a + 1, i_b):
                py, px = path[j]
                if not is_node[py, px]:
                    result[py, px] = False
            a, b = path[i_a], path[i_b]
            rr, cc = _bresenham(a[0], a[1], b[0], b[1])
            keep = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
            result[rr[keep], cc[keep]] = True
        n_fitted += 1
    return result, n_fitted


def snap_to_ring(skel: np.ndarray,
                 cx: float, cy: float, target_r: float,
                 *, max_gap: float = 25.0,
                 tolerance: float = 0.5) -> tuple[np.ndarray, int]:
    """Extend every skeleton endpoint within `max_gap` of the ring `target_r`
    until it reaches that radius. Returns (new_skeleton, n_snapped).

    `target_r` is the radius (from cx, cy) we're trying to *reach*. For sun
    rays terminating just inside the inner ring you want
    target_r = inner_r - inner_stroke / 2 so the extended ray just touches
    the visible ring stroke.

    Walks along the endpoint's branch direction (the direction *away* from
    its single neighbour), not along the radial. For a sun ray that points
    radially these are the same; for a ray that's a few degrees off-radial
    the branch-direction extension stays parallel to the existing ray.
    """
    result = skel.copy()
    endpoints = find_endpoints(skel)
    n_snapped = 0
    for ey, ex in endpoints:
        d = float(np.hypot(ex - cx, ey - cy))
        if abs(target_r - d) > max_gap:
            continue
        dy_unit, dx_unit = _endpoint_direction(skel, int(ey), int(ex))
        if dy_unit == 0.0 and dx_unit == 0.0:
            continue
        # Walk along the branch's natural direction. (Earlier we vetoed
        # walks whose radial dot product had the "wrong" sign relative to
        # the gap, but that was redundant -- proximity to target_r already
        # ensures we're close enough that either direction is short.)
        snapped_this_one = False
        for step in range(1, int(np.ceil(max_gap)) + 1):
            ny = int(round(ey + step * dy_unit))
            nx = int(round(ex + step * dx_unit))
            if not (0 <= ny < skel.shape[0] and 0 <= nx < skel.shape[1]):
                break
            result[ny, nx] = True
            cur = float(np.hypot(nx - cx, ny - cy))
            if abs(cur - target_r) < tolerance:
                snapped_this_one = True
                break
        if snapped_this_one:
            n_snapped += 1
    return result, n_snapped


def restroke(skel: np.ndarray, stroke_width: float) -> np.ndarray:
    """Dilate a skeleton back to a stroked binary mask of the given width.

    Returns a uint8 image: 0 (ink) / 255 (paper), shape == skel.shape.
    """
    radius = max(1, int(round(stroke_width / 2.0)))
    diameter = 2 * radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    dilated = cv2.dilate(skel.astype(np.uint8), kernel, iterations=1)
    return np.where(dilated > 0, np.uint8(0), np.uint8(255))


def skeleton_to_svg_paths(
        skel: np.ndarray,
        *, stroke_width: float,
        simplify_tolerance: float = 1.5,
        min_segment_length: int = 4,
) -> list[str]:
    """Walk every node-to-node skeleton segment and emit it as an SVG
    `<line>` (2-point straight segment) or `<polyline>` (curved / multi-
    vertex) with stroke-width baked in. Each returned string is a complete,
    self-contained SVG element including the stroke attributes.

    Each segment is simplified via Douglas-Peucker (`approximate_polygon`)
    at `simplify_tolerance` pixels. After simplification:
      * 2 points → emit `<line>` -- the cleanest representation for a
        single straight stroke (sun ray, waterfall tick, mountain ridge).
      * 3+ points → emit `<polyline>` -- the medial-axis of a curved
        stroke (sun semicircle, snowcap squiggle).

    Each element gets `fill="none"` and `stroke-linecap="round"` so the
    stroke caps look like the source's pen-drawn caps rather than the
    flat default. Stroke color is set per element to "black" so it
    overrides any parent `<g>` styling that vtracer-style code expected.

    Replaces vtracer's filled-Bezier output for the inside-the-rings
    region. Trades vtracer's wobbly two-edge contours for a single
    centerline element per stroke; the resulting SVG is much smaller
    and edit-friendly.
    """
    is_node = _node_mask(skel)
    H, W = skel.shape
    visited_starts: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    elements: list[str] = []
    stroke_attrs = (f'stroke="black" stroke-width="{stroke_width:.2f}" '
                    f'fill="none" stroke-linecap="round" '
                    f'stroke-linejoin="round"')

    def _emit_segment(path: list[tuple[int, int]]) -> None:
        if len(path) < min_segment_length:
            return
        pts = np.asarray(path, dtype=np.float64)
        simplified = approximate_polygon(pts, tolerance=simplify_tolerance)
        if len(simplified) < 2:
            return
        if len(simplified) == 2:
            (y0, x0), (y1, x1) = simplified[0], simplified[1]
            elements.append(
                f'<line x1="{x0:.2f}" y1="{y0:.2f}" '
                f'x2="{x1:.2f}" y2="{y1:.2f}" {stroke_attrs}/>')
        else:
            pts_str = " ".join(f"{x:.2f},{y:.2f}" for y, x in simplified)
            elements.append(f'<polyline points="{pts_str}" {stroke_attrs}/>')

    node_ys, node_xs = np.where(is_node)
    for sy, sx in zip(node_ys, node_xs):
        start = (int(sy), int(sx))
        for first in _ink_neighbors(skel, start[0], start[1]):
            if (start, first) in visited_starts:
                continue
            path: list[tuple[int, int]] = [start, first]
            prev: tuple[int, int] = start
            cur: tuple[int, int] = first
            steps = 0
            max_steps = H + W
            while not is_node[cur]:
                steps += 1
                if steps > max_steps:
                    break
                forward = _ink_neighbors(skel, cur[0], cur[1], exclude=prev)
                if not forward:
                    break
                prev = cur
                cur = forward[0]
                path.append(cur)
            visited_starts.add((start, first))
            if len(path) >= 2:
                visited_starts.add((path[-1], path[-2]))
            _emit_segment(path)

    return elements
