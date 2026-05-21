"""Phase 3: local sweep over (font, size, letter-spacing, dx/dy, arc radius).

For each text region we already have an initial guess from font-ID. This
module renders that text into the *source crop region* with various
parameter perturbations, picks the parameters that minimize a pixel-level
loss, and returns updated FontMatch values for the assembler to use.

Crucially it sweeps across the top-K finalist fonts simultaneously: weight
is not part of a continuous axis, so we just try several candidate TTFs
and let the loss decide which one survives.
"""
from __future__ import annotations

import math
from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .fontid import _crop_text, _score
from .types import Baseline, DetectedText, FontMatch


# --- text rendering on an arc, in pure PIL ---------------------------------

def _measure_run(text: str, font: ImageFont.FreeTypeFont,
                 letter_spacing_px: float = 0.0) -> tuple[list[tuple[str, int]], int, int, int]:
    """Returns (per-char (ch, advance) list, total_w, ascent, descent)."""
    ascent, descent = font.getmetrics()
    char_advances: list[tuple[str, int]] = []
    total = 0
    for i, ch in enumerate(text):
        # advance = width of bbox; for space use a reasonable fixed advance
        bb = font.getbbox(ch)
        adv = bb[2] - bb[0]
        if ch == " ":
            adv = max(adv, int(font.size * 0.3))
        char_advances.append((ch, max(adv, 0)))
        total += adv
        if i < len(text) - 1:
            total += int(letter_spacing_px)
    return char_advances, total, ascent, descent


def render_arc_text(canvas_h: int, canvas_w: int,
                    text: str, ttf_path: str, size_px: float,
                    cx: float, cy: float, r: float,
                    letter_spacing_px: float = 0.0,
                    dy: float = 0.0,
                    dx: float = 0.0) -> np.ndarray:
    """Render `text` on an arc of radius r centered at (cx, cy + dy) so
    each glyph's baseline sits on the arc, glyph is rotated to the tangent.

    Returns a grayscale uint8 ndarray (white bg, black ink). The text is
    centered on the arc (so it spans equal arc-length to either side of
    angle -pi/2, the top of the circle). `dx` shifts along the arc.
    """
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    cy_eff = cy + dy
    font = ImageFont.truetype(ttf_path, max(int(round(size_px)), 8))
    chars, total_w, ascent, descent = _measure_run(text, font, letter_spacing_px)
    if total_w <= 0:
        return np.array(img)

    # Convert linear total-width into angular total via arc length L = r * dtheta
    total_dtheta = total_w / max(r, 1.0)
    start_theta = -math.pi / 2 - total_dtheta / 2 + dx / max(r, 1.0)

    # Pre-render each glyph individually onto a transparent canvas, rotate, paste.
    cursor = 0
    for ch, adv in chars:
        # angle at the midpoint of this glyph
        glyph_theta_mid = start_theta + (cursor + adv / 2.0) / max(r, 1.0)
        # baseline anchor on the arc
        anchor_x = cx + r * math.cos(glyph_theta_mid)
        anchor_y = cy_eff + r * math.sin(glyph_theta_mid)
        # tangent angle (perpendicular to radius vector)
        tangent_deg = math.degrees(glyph_theta_mid + math.pi / 2)
        # Render this single glyph in a tight box
        bb = font.getbbox(ch)
        gw = max(bb[2] - bb[0], 1)
        gh = max(bb[3] - bb[1], 1)
        # pad a little
        pad = 2
        gimg = Image.new("LA", (gw + 2 * pad, gh + 2 * pad), color=(255, 0))
        gdraw = ImageDraw.Draw(gimg)
        gdraw.text((pad - bb[0], pad - bb[1]), ch, fill=(0, 255), font=font)
        # Rotate so the baseline is parallel to the arc tangent.
        # In PIL the rotation is CCW with the screen Y inverted -> use -tangent_deg.
        rot = gimg.rotate(-tangent_deg, resample=Image.BICUBIC, expand=True)
        # The glyph's anchor is its baseline-left (approx the bottom-left of bbox).
        # In gimg before rotation, the baseline-left is at (pad - bb[0], pad - bb[1] + bb[3]).
        anchor_in_glyph = (pad - bb[0], pad - bb[1] + bb[3])
        # Compute where that anchor lands after rotation (rotate around center).
        cx_g, cy_g = gimg.size[0] / 2, gimg.size[1] / 2
        # Vector from rotation center to anchor:
        vx = anchor_in_glyph[0] - cx_g
        vy = anchor_in_glyph[1] - cy_g
        # After rotating by -tangent_deg around the glyph center
        c = math.cos(math.radians(-tangent_deg))
        s = math.sin(math.radians(-tangent_deg))
        rvx = c * vx - s * vy
        rvy = s * vx + c * vy
        # Now the anchor is at (rot.center + (rvx, rvy)).
        rcx, rcy = rot.size[0] / 2, rot.size[1] / 2
        anchor_in_rot = (rcx + rvx, rcy + rvy)
        # Paste so the anchor aligns with (anchor_x, anchor_y)
        paste_x = int(round(anchor_x - anchor_in_rot[0]))
        paste_y = int(round(anchor_y - anchor_in_rot[1]))
        img.paste(rot, (paste_x, paste_y), rot)
        cursor += adv + int(letter_spacing_px)

    return np.array(img)


def render_line_text(canvas_h: int, canvas_w: int,
                     text: str, ttf_path: str, size_px: float,
                     x: float, y: float,
                     letter_spacing_px: float = 0.0) -> np.ndarray:
    """Render straight text centered at (x, y), baseline through y."""
    img = Image.new("L", (canvas_w, canvas_h), color=255)
    font = ImageFont.truetype(ttf_path, max(int(round(size_px)), 8))
    chars, total_w, _, _ = _measure_run(text, font, letter_spacing_px)
    cursor = x - total_w / 2.0
    bb_y = 0
    # use first non-empty char's bbox to fix baseline -> top offset
    for ch, _ in chars:
        if ch.strip():
            bb_y = font.getbbox(ch)[3]
            break
    draw = ImageDraw.Draw(img)
    for ch, adv in chars:
        bb = font.getbbox(ch)
        draw.text((cursor - bb[0], y - bb[3]), ch, fill=0, font=font)
        cursor += adv + letter_spacing_px
    return np.array(img)


# --- scoring -------------------------------------------------------------

def _crop_with_mask(image: np.ndarray, polygon: np.ndarray,
                    pad: int | None = None,
                    pad_factor: float = 0.5
                    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Axis-aligned crop around a polygon, padded by `pad_factor * polygon_size`.

    A generous padding (default 50% of each dim) ensures the loss penalises
    rendered text that extends beyond the source's bbox -- otherwise the
    optimizer learns to pick huge fonts whose extra ink falls outside the
    crop and so isn't scored.
    """
    H, W = image.shape[:2]
    poly_w = float(polygon[:, 0].max() - polygon[:, 0].min())
    poly_h = float(polygon[:, 1].max() - polygon[:, 1].min())
    if pad is None:
        pad_x = int(poly_w * pad_factor)
        pad_y = int(poly_h * pad_factor)
    else:
        pad_x = pad_y = pad
    x0 = max(int(polygon[:, 0].min()) - pad_x, 0)
    x1 = min(int(polygon[:, 0].max()) + pad_x + 1, W)
    y0 = max(int(polygon[:, 1].min()) - pad_y, 0)
    y1 = min(int(polygon[:, 1].max()) + pad_y + 1, H)
    return image[y0:y1, x0:x1], (x0, y0, x1, y1)


def _eval_params_arc(text: str, ttf_path: str, size_px: float,
                     baseline: Baseline, dy: float, dx: float, ls: float,
                     source_gray: np.ndarray, viewport: tuple[int, int, int, int],
                     full_h: int, full_w: int,
                     max_text_w: float = float("inf")) -> float:
    """Render the text on its arc inside `viewport` and return composite loss."""
    # Cheap pre-check: if the rendered string would overflow `max_text_w`, drop
    # immediately rather than render and find out the canvas truncated it.
    try:
        f = ImageFont.truetype(ttf_path, max(int(round(size_px)), 8))
        _, total_w, _, _ = _measure_run(text, f, ls)
        if total_w > max_text_w:
            return float("inf")
    except Exception:
        return float("inf")
    cx, cy, r, _t0, _t1 = baseline.params
    rendered = render_arc_text(full_h, full_w, text, ttf_path, size_px,
                               cx, cy, r, letter_spacing_px=ls, dy=dy, dx=dx)
    x0, y0, x1, y1 = viewport
    sub = rendered[y0:y1, x0:x1]
    composite, _l1, _v = _score(sub, source_gray)
    return composite


def _eval_params_line(text: str, ttf_path: str, size_px: float,
                      x: float, y: float, ls: float,
                      source_gray: np.ndarray, viewport: tuple[int, int, int, int],
                      full_h: int, full_w: int,
                      max_text_w: float = float("inf")) -> float:
    try:
        f = ImageFont.truetype(ttf_path, max(int(round(size_px)), 8))
        _, total_w, _, _ = _measure_run(text, f, ls)
        if total_w > max_text_w:
            return float("inf")
    except Exception:
        return float("inf")
    rendered = render_line_text(full_h, full_w, text, ttf_path, size_px,
                                x, y, letter_spacing_px=ls)
    x0, y0, x1, y1 = viewport
    sub = rendered[y0:y1, x0:x1]
    composite, _l1, _v = _score(sub, source_gray)
    return composite


# --- sweep --------------------------------------------------------------

def sweep(dt: DetectedText, candidates: list[FontMatch],
          source_img_bgr: np.ndarray,
          *, n_size: int = 5, n_dy: int = 5, n_ls: int = 3,
          n_dx: int = 5,
          try_arc_for_line: bool = True,
          force_arc: bool = False,
          arc_centers_hint: list[tuple[float, float]] | None = None,
          ) -> tuple[FontMatch, dict, Baseline | None]:
    """Run a coordinate-descent grid sweep on each candidate font and
    return (best_font, best_params, best_baseline). best_baseline may
    override dt.baseline if the sweep finds an arc fit beats line.
    """
    if not candidates:
        return FontMatch(family="sans-serif", weight=700), {}, None

    H, W = source_img_bgr.shape[:2]
    source_gray = cv2.cvtColor(source_img_bgr, cv2.COLOR_BGR2GRAY)

    # Restrict the comparison viewport to a polygon bbox padded by 50% of
    # its own dimensions. Generous padding makes "rendered text spills past
    # the source's bbox" visible to the loss; without it the chamfer is
    # silently low because spilled ink simply falls outside the crop.
    crop, viewport = _crop_with_mask(source_gray, dt.polygon, pad_factor=0.5)
    if crop.size == 0:
        return candidates[0], {}, None

    initial_size = candidates[0].size_px if candidates[0].size_px else 64.0
    # Tightened so the search can't pick a font 30% taller than the source
    # by relying on letter-spacing to fill the width.
    size_range = (initial_size * 0.80, initial_size * 1.10)
    sizes = np.linspace(size_range[0], size_range[1], n_size)
    lss = np.linspace(0, initial_size * 0.12, n_ls)

    # Build a set of candidate baselines to try.
    candidate_baselines: list[Baseline] = []
    if dt.baseline is not None and dt.baseline.kind == "arc":
        # The merger / baseline fit gave us a real arc -- trust it,
        # don't synthesize alternates that might displace the text.
        candidate_baselines.append(dt.baseline)
    elif dt.baseline is None or dt.baseline.kind == "line":
        if dt.baseline is not None and not force_arc:
            candidate_baselines.append(dt.baseline)
        cx_p = float(dt.polygon[:, 0].mean())
        cy_p = float(dt.polygon[:, 1].mean())
        text_w = float(dt.polygon[:, 0].max() - dt.polygon[:, 0].min())
        if arc_centers_hint:
            # The caller gave us known arc centers (e.g. the badge's). Sweep
            # only the radius from that center to the polygon centroid, +-
            # a small offset, so the arc stays concentric with the badge.
            for hcx, hcy in arc_centers_hint:
                r_baseline = float(np.hypot(cx_p - hcx, cy_p - hcy))
                for r_mult in (0.85, 0.92, 1.0, 1.08, 1.15):
                    r_arc = r_baseline * r_mult
                    candidate_baselines.append(
                        Baseline(kind="arc",
                                 params=(float(hcx), float(hcy), r_arc, 0.0, 0.0),
                                 residual=0.0)
                    )
        if try_arc_for_line or force_arc:
            # Fallback: synth a few candidate concave-up arcs whose center
            # sits below the polygon. Used when no badge hint is available.
            ks = (1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0) if force_arc else (1.5, 2.5, 4.0, 8.0)
            for k in ks:
                cy_arc = cy_p + k * text_w / 2
                r_arc = abs(cy_arc - cy_p)
                candidate_baselines.append(
                    Baseline(kind="arc",
                             params=(cx_p, cy_arc, r_arc, 0.0, 0.0),
                             residual=0.0)
                )

    text = dt.text or ""

    # Hard upper bound on rendered text width: ~115% of the source polygon's
    # x-extent (or arc chord length for arc baselines). Forbids the optimizer
    # picking a font/size combo whose rendered text overflows -- without this,
    # overflow that falls past the canvas edge isn't scored and the loss
    # silently rewards huge fonts.
    poly_w = float(dt.polygon[:, 0].max() - dt.polygon[:, 0].min())
    max_text_w = poly_w * 1.15

    def render_with(baseline: Baseline, ttf: str, s: float, dy: float,
                    dx: float, ls: float) -> float:
        if baseline.kind == "arc":
            return _eval_params_arc(text, ttf, s, baseline, dy, dx, ls,
                                    crop, viewport, H, W,
                                    max_text_w=max_text_w)
        line_x = float(dt.polygon[:, 0].mean())
        line_y = float(dt.polygon[:, 1].max()) - s * 0.18
        return _eval_params_line(text, ttf, s, line_x + dx, line_y + dy, ls,
                                 crop, viewport, H, W,
                                 max_text_w=max_text_w)

    # Per-baseline displacement ranges
    def ranges(baseline: Baseline) -> tuple[np.ndarray, np.ndarray]:
        if baseline.kind == "arc":
            r0 = baseline.params[2]
            dys_b = np.linspace(-r0 * 0.05, r0 * 0.05, n_dy)
            dxs_b = np.linspace(-initial_size * 0.6, initial_size * 0.6, n_dx)
        else:
            dys_b = np.linspace(-initial_size * 0.3, initial_size * 0.3, n_dy)
            dxs_b = np.linspace(-initial_size * 0.3, initial_size * 0.3, n_dx)
        return dys_b, dxs_b

    best = (float("inf"), None, None, None)  # (score, fm, params, baseline)

    for fm in candidates:
        for baseline in candidate_baselines:
            dys_b, dxs_b = ranges(baseline)
            local_best = (float("inf"), None)
            # Coarse: sweep (size, dy, ls) at dx=0
            for s in sizes:
                for dy in dys_b:
                    for ls in lss:
                        sc = render_with(baseline, fm.ttf_path, float(s),
                                          float(dy), 0.0, float(ls))
                        if sc < local_best[0]:
                            local_best = (sc, dict(size_px=float(s),
                                                   dy=float(dy),
                                                   ls_px=float(ls),
                                                   dx=0.0))
            # If nothing produced a finite score, skip this (font, baseline).
            if local_best[1] is None:
                continue
            # Refine dx around local best
            for dx in dxs_b:
                params = dict(local_best[1])
                params["dx"] = float(dx)
                sc = render_with(baseline, fm.ttf_path, params["size_px"],
                                  params["dy"], params["dx"], params["ls_px"])
                if sc < local_best[0]:
                    local_best = (sc, params)
            if local_best[0] < best[0]:
                best = (local_best[0], fm, local_best[1], baseline)

    final_score, fm, params, baseline = best
    if fm is None:
        return candidates[0], {}, None
    out = replace(fm,
                  size_px=params["size_px"],
                  letter_spacing_em=params["ls_px"] / max(params["size_px"], 1.0),
                  dx=params["dx"], dy=params["dy"],
                  score=final_score)
    # If the chosen baseline is a synthesized arc (t0==t1==0), refill its
    # angular extent from the chosen font's actual rendered width so the
    # assembler can draw a non-degenerate path.
    new_baseline = baseline if baseline is not dt.baseline else None
    if new_baseline is not None and new_baseline.kind == "arc":
        cx, cy, r, t0, t1 = new_baseline.params
        if t0 == 0.0 and t1 == 0.0:
            font = ImageFont.truetype(fm.ttf_path,
                                      max(int(round(params["size_px"])), 8))
            _chars, total_w, _, _ = _measure_run(text, font, params["ls_px"])
            total_dtheta = total_w / max(r, 1.0)
            # Centered on the top of the circle (-pi/2). Apply dx along the arc.
            center_theta = -math.pi / 2 + params["dx"] / max(r, 1.0)
            new_t0 = center_theta - total_dtheta / 2
            new_t1 = center_theta + total_dtheta / 2
            new_baseline = Baseline(kind="arc",
                                    params=(cx, cy, r, float(new_t0), float(new_t1)),
                                    residual=new_baseline.residual)
    return out, params, new_baseline
