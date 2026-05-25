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


def refit_arc_from_letters(img_bgr: np.ndarray,
                           polygon: np.ndarray,
                           baseline: Baseline | None) -> Baseline | None:
    """Refit a text polygon's arc baseline through the centroids of the
    individual letters detected inside the polygon.

    The polygon's own vertices typically lie on the OUTER edge of the text
    band, not on the baseline — so a CircleModel fit through polygon
    vertices gives an arc that's offset from where the letters actually
    sit. The optimal arc is the one that runs through the letter centroids,
    which can differ from the badge center by 10-20 px (enough that text
    placed on a badge-centric arc visibly drifts at the arc's extremes).

    Returns a new Baseline(kind="arc", ...) on success, or the input
    baseline unchanged on any failure.
    """
    if polygon is None or len(polygon) < 3:
        return baseline

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    # Connected components on the binarized image. Filter to those whose
    # centroid sits inside (or just outside) the polygon. The polygon may be
    # tight against the text bbox; allow up to 30 px slack.
    binary = (gray < 128).astype(np.uint8) * 255
    n_labels, _labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    poly_f = polygon.astype(np.float32)
    centroids: list[tuple[float, float]] = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 50 or area > 20000:
            continue
        cx_l, cy_l = float(cents[i, 0]), float(cents[i, 1])
        if cv2.pointPolygonTest(poly_f, (cx_l, cy_l), True) < -30:
            continue
        centroids.append((cx_l, cy_l))

    if len(centroids) < 3:
        return baseline

    # Group fragments at the same x (rare in this corpus but happens for
    # split glyphs like "5") so each letter contributes one centroid.
    centroids.sort(key=lambda c: c[0])
    grouped: list[tuple[float, float]] = []
    typical_w = float(np.median([stats[i, 2] for i in range(1, n_labels) if stats[i, 4] > 200]))
    merge_threshold = typical_w * 0.7 if typical_w > 0 else 30.0
    cur = [centroids[0]]
    for c in centroids[1:]:
        if c[0] - cur[-1][0] <= merge_threshold:
            cur.append(c)
        else:
            grouped.append((float(np.mean([p[0] for p in cur])),
                            float(np.mean([p[1] for p in cur]))))
            cur = [c]
    grouped.append((float(np.mean([p[0] for p in cur])),
                    float(np.mean([p[1] for p in cur]))))

    # Need ≥6 letter centroids for a stable circle fit. With fewer points
    # (e.g. "2025" with only 4 digits forming a shallow arc) the fit is
    # numerically degenerate — small noise in any point swings the center
    # by hundreds of pixels.
    if len(grouped) < 6:
        return baseline

    pts = np.array(grouped, dtype=np.float64)
    xs, ys = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * xs, 2 * ys, -np.ones(len(xs))])
    b = xs ** 2 + ys ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx_fit, cy_fit, D = sol
    r2 = cx_fit ** 2 + cy_fit ** 2 - D
    if r2 <= 0:
        return baseline
    r_fit = float(np.sqrt(r2))
    # Sanity: reject absurd fits (e.g. nearly-collinear centroids producing
    # huge circles), which can happen for short text like "2025" with only
    # 4 points along a shallow arc.
    poly_span = float(max(np.ptp(xs), np.ptp(ys)))
    if r_fit > 10 * poly_span:
        return baseline
    residual = float(np.mean(np.abs(np.sqrt((xs - cx_fit) ** 2 + (ys - cy_fit) ** 2) - r_fit)))
    t0 = math.atan2(ys[0] - cy_fit, xs[0] - cx_fit)
    t1 = math.atan2(ys[-1] - cy_fit, xs[-1] - cx_fit)
    return Baseline(
        kind="arc",
        params=(float(cx_fit), float(cy_fit), r_fit, float(t0), float(t1)),
        residual=residual,
    )


def sweep_char_anchor(img_bgr: np.ndarray, ttf_path: str, size_px: float,
                      char: str, anchor_x: float, anchor_y: float,
                      rot_deg: float, mask: np.ndarray,
                      search_radius: float = 25.0,
                      rot_search_deg: float = 0.0,
                      size_search_pct: float = 0.0,
                      ) -> tuple[float, float, float, float]:
    """Sweep (dx, dy, optional rotation delta) to maximize IoU of a
    rendered character against the original ink inside `mask`. Returns
    (best_x, best_y, best_rot_deg).

    Set rot_search_deg > 0 to ALSO sweep ± that many degrees of rotation
    around the supplied `rot_deg`. The tangent angle from the arc is
    usually correct but per-letter optical adjustments in the original
    can be ±2° off."""
    from PIL import Image as _Image, ImageDraw as _ImageDraw, ImageFont as _ImageFont

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    orig_ink = gray < 128
    H, W = gray.shape

    def render_rotated(rot, size):
        font = _ImageFont.truetype(ttf_path, int(round(size)))
        bb = font.getbbox(char)
        gw_l = bb[2] - bb[0]; gh_l = bb[3] - bb[1]
        cs_l = int(max(gw_l, gh_l) * 2.5 + 40)
        canvas = _Image.new("L", (cs_l, cs_l), 255)
        draw = _ImageDraw.Draw(canvas)
        tx = cs_l / 2 - (bb[0] + gw_l / 2)
        ty = cs_l / 2 - (bb[1] + gh_l / 2)
        draw.text((tx, ty), char, fill=0, font=font)
        return np.array(canvas.rotate(-rot, fillcolor=255, resample=_Image.BICUBIC)), cs_l

    def iou_at(rotated, cs_l, dx, dy):
        full = np.full((H, W), 255, dtype=np.uint8)
        px = int(round(anchor_x + dx - cs_l / 2))
        py = int(round(anchor_y + dy - cs_l / 2))
        xs0 = max(0, -px); ys0 = max(0, -py)
        xs1 = min(cs_l, W - px); ys1 = min(cs_l, H - py)
        if xs1 <= xs0 or ys1 <= ys0:
            return 0.0
        src = rotated[ys0:ys1, xs0:xs1]
        dx0 = max(0, px); dy0 = max(0, py)
        full[dy0:dy0 + (ys1 - ys0), dx0:dx0 + (xs1 - xs0)] = np.minimum(
            full[dy0:dy0 + (ys1 - ys0), dx0:dx0 + (xs1 - xs0)], src)
        rend_ink = full < 128
        a = orig_ink & mask; b = rend_ink & mask
        inter = (a & b).sum()
        union = (a | b).sum()
        return inter / max(union, 1)

    rot_candidates = ([rot_deg + d for d in np.linspace(-rot_search_deg, rot_search_deg, 5)]
                      if rot_search_deg > 0 else [rot_deg])
    size_candidates = ([size_px * (1 + p) for p in np.linspace(-size_search_pct, size_search_pct, 5)]
                       if size_search_pct > 0 else [size_px])

    best_iou = -1.0
    best_dx = best_dy = 0.0
    best_rot = rot_deg
    best_size = size_px
    # Coarse search across (rotation, size, dx, dy) at ~2.5px step.
    for rot in rot_candidates:
        for sz in size_candidates:
            rotated, cs_l = render_rotated(rot, sz)
            for dx in np.linspace(-search_radius, search_radius, 9):
                for dy in np.linspace(-search_radius, search_radius, 9):
                    s = iou_at(rotated, cs_l, dx, dy)
                    if s > best_iou:
                        best_iou = s
                        best_dx = float(dx); best_dy = float(dy)
                        best_rot = float(rot); best_size = float(sz)
    # Fine pass: ±step around coarse winner at 7x7 = ~0.7px step.
    rotated, cs_l = render_rotated(best_rot, best_size)
    step = (2 * search_radius) / 8
    for dx in np.linspace(best_dx - step, best_dx + step, 7):
        for dy in np.linspace(best_dy - step, best_dy + step, 7):
            s = iou_at(rotated, cs_l, dx, dy)
            if s > best_iou:
                best_iou = s; best_dx = float(dx); best_dy = float(dy)
    # Subpixel pass: ±0.6 px at 0.15px step.
    for dx in np.linspace(best_dx - 0.6, best_dx + 0.6, 9):
        for dy in np.linspace(best_dy - 0.6, best_dy + 0.6, 9):
            s = iou_at(rotated, cs_l, dx, dy)
            if s > best_iou:
                best_iou = s; best_dx = float(dx); best_dy = float(dy)
    return anchor_x + best_dx, anchor_y + best_dy, best_rot, best_size


def detect_letter_anchors(
        img_bgr: np.ndarray,
        polygon: np.ndarray,
        text: str,
        *, badge_center: tuple[float, float] = (0.0, 0.0),
        slack_px: float = 30.0,
        ) -> list[dict] | None:
    """Detect per-character anchors inside a text polygon by running
    connected components on the binarized image and grouping fragments by
    arc-length along the polygon's baseline.

    Returns a list of {x, y, rot_deg, cap_h} dicts ordered to match
    `text` (spaces skipped), or None on failure / mismatched count.

    `badge_center` decides top-vs-bottom arc orientation (used for the
    rotation sign so letters stand upright).
    """
    if polygon is None or len(polygon) < 3 or not text:
        return None
    expected = [c for c in text if c != " "]
    if not expected:
        return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    binary = (gray < 128).astype(np.uint8) * 255
    n_labels, _labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    poly_f = polygon.astype(np.float32)

    fragments = []
    for i in range(1, n_labels):
        x, y, w, h, area = stats[i]
        if area < 50 or area > 20000:
            continue
        cx_l, cy_l = float(cents[i, 0]), float(cents[i, 1])
        if cv2.pointPolygonTest(poly_f, (cx_l, cy_l), True) < -slack_px:
            continue
        fragments.append({
            "i": i,
            "centroid": (cx_l, cy_l),
            "bbox": (x, y, w, h),
            "area": area,
        })
    if len(fragments) < len(expected):
        return None

    bc_x, bc_y = badge_center
    cs = np.array([f["centroid"] for f in fragments])
    is_top = cs[:, 1].mean() < bc_y

    # Sort fragments by signed arc-length from arc center. We use the
    # BADGE center here (not a per-text refit) because we want a stable
    # ordering reference; the actual rotation per letter uses the same
    # center, which is close enough for ordering.
    def pos(f):
        cx, cy = f["centroid"]
        theta = math.atan2(cy - bc_y, cx - bc_x)
        r = math.hypot(cx - bc_x, cy - bc_y)
        return (1 if is_top else -1) * theta * r
    fragments.sort(key=pos)

    big_widths = [f["bbox"][2] for f in fragments if f["area"] > 200]
    if not big_widths:
        return None
    typical_w = float(np.median(big_widths))
    merge_threshold = typical_w * 0.7

    groups = [[fragments[0]]]
    for f in fragments[1:]:
        if pos(f) - pos(groups[-1][-1]) <= merge_threshold:
            groups[-1].append(f)
        else:
            groups.append([f])

    if len(groups) != len(expected):
        # Count mismatch — fall back to textPath rather than risk misaligned
        # per-letter placement.
        return None

    # Build per-letter dilated masks. Fragments already store their CC
    # label index in f["i"]; use that directly instead of re-looking-up
    # via centroid (which fails for letters whose centroid is INSIDE the
    # glyph's counter, like C, O, P — that pixel has label 0 = background).
    _, cc_labels, _, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    group_masks: list[np.ndarray] = []
    for g in groups:
        m = np.zeros(gray.shape, dtype=np.uint8)
        for f in g:
            m |= ((cc_labels == f["i"]).astype(np.uint8) * 255)
        group_masks.append(cv2.dilate(m, kernel) > 0)

    raw = []
    for ch, g in zip(expected, groups):
        xs0 = min(f["bbox"][0] for f in g)
        ys0 = min(f["bbox"][1] for f in g)
        xs1 = max(f["bbox"][0] + f["bbox"][2] for f in g)
        ys1 = max(f["bbox"][1] + f["bbox"][3] for f in g)
        total_area = sum(f["area"] for f in g)
        cx_g = sum(f["centroid"][0] * f["area"] for f in g) / total_area
        cy_g = sum(f["centroid"][1] * f["area"] for f in g) / total_area
        anchor_x = (xs0 + xs1) / 2.0
        anchor_y = (ys0 + ys1) / 2.0
        raw.append({
            "char": ch, "anchor_x": anchor_x, "anchor_y": anchor_y,
            "cap_h": ys1 - ys0,
        })

    # Fit a circle through the RAW per-letter bbox centers. This finds the
    # ACTUAL text arc center, which may be offset from the badge center
    # by 10-20 px (the designer often places text on an arc whose center
    # is not identical to the ring center). With ≥6 points the fit is
    # stable; below that the result is degenerate and we fall back to
    # snapping around the badge center.
    xs_arr = np.array([r["anchor_x"] for r in raw])
    ys_arr = np.array([r["anchor_y"] for r in raw])
    fit_cx, fit_cy = bc_x, bc_y
    fit_r: float | None = None
    if len(raw) >= 6:
        A_mat = np.column_stack([2*xs_arr, 2*ys_arr, -np.ones(len(xs_arr))])
        b_vec = xs_arr**2 + ys_arr**2
        sol, *_ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
        cx_f, cy_f, D = sol
        r2 = cx_f**2 + cy_f**2 - D
        # Sanity: reject absurd fits.
        poly_span = float(max(np.ptp(xs_arr), np.ptp(ys_arr)))
        if r2 > 0 and float(np.sqrt(r2)) <= 10 * poly_span:
            fit_cx, fit_cy = float(cx_f), float(cy_f)
            fit_r = float(np.sqrt(r2))

    # Snap each anchor onto the fitted circle. With the right CENTER, the
    # snapped positions match the designer's intended layout — A's
    # bbox-center radial mismatch disappears.
    if fit_r is None:
        radii = [math.hypot(r["anchor_x"] - fit_cx, r["anchor_y"] - fit_cy)
                 for r in raw]
        fit_r = float(np.median(radii))

    anchors: list[dict] = []
    for i, r_info in enumerate(raw):
        dx_l = r_info["anchor_x"] - fit_cx
        dy_l = r_info["anchor_y"] - fit_cy
        d = math.hypot(dx_l, dy_l) or 1.0
        snap_x = fit_cx + dx_l * fit_r / d
        snap_y = fit_cy + dy_l * fit_r / d
        theta = math.atan2(dy_l, dx_l)
        rot_deg = math.degrees(theta) + (90.0 if is_top else -90.0)
        anchors.append({
            "char": r_info["char"],
            "x": float(snap_x),
            "y": float(snap_y),
            "rot_deg": float(rot_deg),
            "cap_h": float(r_info["cap_h"]),
            # Attach fitted arc params on every anchor — caller can read
            # them off the first entry to set the baseline.
            "arc_cx": float(fit_cx),
            "arc_cy": float(fit_cy),
            "arc_r": float(fit_r),
            # Bbox-center (pre-snap) for endpoint sweeping.
            "anchor_x_raw": float(r_info["anchor_x"]),
            "anchor_y_raw": float(r_info["anchor_y"]),
            "mask": group_masks[i],
        })
    return anchors


def place_text_on_arc(text: str, ttf_path: str, size_px: float,
                      arc_cx: float, arc_cy: float, arc_r: float,
                      t_center: float, is_top: bool = True,
                      letter_spacing_px: float = 0.0,
                      ) -> list[dict]:
    """Algorithmic per-character placement on a circular arc.

    Computes per-character (x, y, rot_deg) by:
      1. Asking the font for each character's ADVANCE width.
      2. Centering the cumulative text width on `t_center`.
      3. Mapping each character's center-x (in unwrapped text-line coords)
         to an angular position on the arc.
      4. Rotating each character by the tangent angle at its arc position.

    Works for arbitrary text (different lengths, different fonts) since the
    advances come from the font itself. For matching badges with manually-
    tweaked letter spacing, this approximates the designer's intent without
    hard-coding per-character offsets.
    """
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(ttf_path, int(round(size_px)))
    except Exception:
        return []

    # Cumulative advance positions (text width up to and including i-th char).
    # Apply uniform letter_spacing_px between consecutive characters so the
    # algorithm can match hand-set designs whose tracking is wider than
    # the font's natural advance width.
    cumulative = [0.0]
    for i in range(len(text)):
        natural = float(font.getlength(text[: i + 1]))
        cumulative.append(natural + letter_spacing_px * i)
    total_w = cumulative[-1]
    if total_w <= 0:
        return []

    # Each character's center-x in unwrapped coords.
    char_centers = [(cumulative[i] + cumulative[i + 1]) / 2 for i in range(len(text))]
    total_theta = total_w / arc_r
    # Direction of arc traversal as a function of unwrapped text x:
    #   top text: theta INCREASES from leftmost (small theta) to rightmost.
    #   bottom text: theta DECREASES — leftmost letter has the LARGER
    #   theta because for points below the arc center, atan2(dy, dx) flips
    #   sign as dx crosses 0 from negative to positive.
    # So the "anchor" angle for unwrap_x=0 is on the opposite side of
    # t_center for bottom vs top text, and the increment per pixel is
    # signed accordingly.
    direction = 1 if is_top else -1
    t0 = t_center - direction * (total_theta / 2.0)

    placements: list[dict] = []
    for ch, cx_unwrap in zip(text, char_centers):
        if ch == " ":
            continue
        theta = t0 + direction * cx_unwrap / arc_r
        x = arc_cx + arc_r * math.cos(theta)
        y = arc_cy + arc_r * math.sin(theta)
        rot_deg = math.degrees(theta) + (90.0 if is_top else -90.0)
        placements.append({
            "char": ch,
            "x": float(x), "y": float(y),
            "rot_deg": float(rot_deg),
        })
    return placements


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
