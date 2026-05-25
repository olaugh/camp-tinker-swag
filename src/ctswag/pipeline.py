"""End-to-end pipeline runner."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from . import baseline as baseline_mod
from . import circles as circles_mod
from . import detect as detect_mod
from . import fontid as fontid_mod
from . import geom as geom_mod
from . import inpaint as inpaint_mod
from . import merge as merge_mod
from . import optimize as optimize_mod
from . import recognize as recognize_mod
from . import skeleton as skeleton_mod
from . import snap_strokes as snap_strokes_mod
from . import trace as trace_mod
from . import unwarp as unwarp_mod
from .assemble import assemble
from .types import Baseline, DetectedText, FontMatch, PipelineResult, TraceResult


@dataclass
class PipelineConfig:
    detector: str = "synth"
    recognizer: str = "tesseract"
    inpainter: str = "white"
    tracer: str = "vtracer"
    font_corpus_dir: str = "fonts"
    text_height_factor: float = 1.4   # crop height multiplier vs polygon height
    detector_kwargs: dict[str, Any] = field(default_factory=dict)
    tracer_kwargs: dict[str, Any] = field(default_factory=dict)
    top_k_fonts: int = 5
    refine: bool = True               # Run optimize.sweep on top-K finalists
    refine_top_k: int = 6             # How many finalists to keep into the sweep
    refine_try_arc: bool = True       # For "line" baselines, also sweep arc radii
    refine_force_arc: bool = False    # Force every text to use an arc baseline
    corpus_min_weight: int = 0        # Drop TTFs lighter than this (700=bold)
    detect_circles: bool = True       # HoughCircles ring detection -> <circle>
    # Off by default: the skeleton path replaces vtracer's filled-Bezier
    # outlines with single-centerline <line>/<polyline> elements per stroke,
    # which fixes specific vtracer artifacts (e.g. waterfall ticks getting
    # rendered as inverted "T"s where they brush the mountain baseline) but
    # introduces discontinuities at junctions and faceted curves where the
    # source had smooth ones (sun semicircle). Enable when you specifically
    # want the cleaner per-stroke SVG structure; vtracer is the default
    # because its raster fidelity is currently better.
    use_skeleton_for_inside: bool = False
    # Post-vtracer: for paths whose bbox is below the mountain-baseline
    # horizon AND are tall-narrow AND whose PCA major-axis is within this
    # many degrees of vertical, snap to an exactly-vertical <line>. Picks
    # up the rounded-tip waterfall ticks (the source draws them strictly
    # vertical but vtracer's trace lands a few degrees off). Set to 0 to
    # disable. Only applies to inside-the-rings paths.
    vertical_snap_tolerance_deg: float = 8.0
    # If provided, skip OCR entirely and assign these strings to detected
    # polygons in top-to-bottom order. Same length as expected polygon count.
    text_overrides: list[str] | None = None
    # If provided, skip font-ID for the i-th text and use this (family, weight,
    # ttf_path) directly. Same top-to-bottom ordering as text_overrides.
    # None entries fall through to font-ID. Useful for debugging a specific
    # font hypothesis side-by-side with the auto-picked answer.
    font_overrides: list[tuple[str, int, str] | None] | None = None


def _crop_polygon(img: np.ndarray, polygon: np.ndarray,
                  *, height_mul: float = 1.0) -> np.ndarray:
    """Tight bbox crop. (For arched text we crop the bbox of the polygon.)"""
    x0, y0 = polygon.min(axis=0)
    x1, y1 = polygon.max(axis=0)
    H, W = img.shape[:2]
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    h = (y1 - y0) * height_mul
    w = (x1 - x0)
    x0 = int(max(0, cx - w/2))
    x1 = int(min(W, cx + w/2))
    y0 = int(max(0, cy - h/2))
    y1 = int(min(H, cy + h/2))
    return img[y0:y1, x0:x1].copy()


def run(input_path: Path | str,
        output_path: Path | str,
        cfg: PipelineConfig,
        *,
        badge=None) -> PipelineResult:
    """Run a full configuration end-to-end. Returns a PipelineResult."""
    timings: dict[str, float] = {}
    config_dump = {
        "detector": cfg.detector, "recognizer": cfg.recognizer,
        "inpainter": cfg.inpainter, "tracer": cfg.tracer,
        "detector_kwargs": cfg.detector_kwargs,
        "tracer_kwargs": cfg.tracer_kwargs,
    }
    out = PipelineResult(input_path=str(input_path), output_svg_path=str(output_path),
                         timings=timings, config=config_dump)

    t0 = time.perf_counter()
    img_bgr = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(input_path)
    H, W = img_bgr.shape[:2]
    timings["load"] = time.perf_counter() - t0

    # 1. detect
    t0 = time.perf_counter()
    det_fn = detect_mod.REGISTRY[cfg.detector]
    kw = dict(cfg.detector_kwargs)
    if cfg.detector == "synth":
        kw["badge"] = badge
    polygons = det_fn(img_bgr, **kw)
    timings["detect"] = time.perf_counter() - t0

    # 1b. detect rings early so we can use the badge geometry to measure
    # text bands. Detector polygons (camp_geom) use hardcoded radius/height
    # ratios; the actual text in the source sits at radii we can simply
    # measure once we know the outer ring's position. This call's result is
    # cached in `found_circles` / `circle_widths` and reused at the inpaint
    # stage below -- no duplicate detection.
    found_circles: list[tuple[float, float, float]] = []
    circle_widths: list[float] = []
    if cfg.detect_circles:
        t0 = time.perf_counter()
        rings = circles_mod.detect_with_strokes(img_bgr)
        found_circles = [(cx, cy, r) for (cx, cy, r, _w) in rings]
        circle_widths = [w for (_cx, _cy, _r, w) in rings]
        timings["circles"] = time.perf_counter() - t0
        out.config["circles"] = [
            {"cx": cx, "cy": cy, "r": r, "stroke_width": w}
            for (cx, cy, r), w in zip(found_circles, circle_widths)
        ]

    # 1c. measure text bands and replace detector polygons with the actual
    # arc-shaped bands the text occupies. Each polygon is matched to a
    # measured band by angular centroid; polygons with no nearby band keep
    # the detector's geometry as a fallback.
    if found_circles:
        outer_cx, outer_cy, outer_r = found_circles[0]
        outer_sw = circle_widths[0]
        bands = geom_mod.measure_text_bands(
            img_bgr, outer_cx, outer_cy, outer_r, outer_sw)
        measured_baselines: list[Baseline | None] = [None] * len(polygons)
        if bands and polygons:
            import math as _math
            new_polygons: list[np.ndarray] = []
            for i, poly in enumerate(polygons):
                py_cx = float(poly[:, 0].mean())
                py_cy = float(poly[:, 1].mean())
                poly_theta = _math.atan2(py_cy - outer_cy, py_cx - outer_cx)
                best_band: dict | None = None
                best_dist = _math.inf
                for band in bands:
                    band_mid = (band["theta_min"] + band["theta_max"]) / 2.0
                    d = ((poly_theta - band_mid + _math.pi)
                         % (2 * _math.pi)) - _math.pi
                    if abs(d) < best_dist:
                        best_dist = abs(d)
                        best_band = band
                if best_band is not None and best_dist < _math.radians(30.0):
                    new_polygons.append(geom_mod.band_to_polygon(best_band))
                    measured_baselines[i] = geom_mod.band_to_baseline(best_band)
                else:
                    new_polygons.append(poly)
            polygons = new_polygons
        out.config["text_bands"] = bands
    else:
        measured_baselines = [None] * len(polygons)

    # 2. recognize + baseline + crop, per polygon
    t0 = time.perf_counter()
    pre_texts: list[str] = []
    pre_polys: list[np.ndarray] = []

    if cfg.text_overrides is not None:
        # Skip OCR entirely. We need a label per detected polygon to feed the
        # arc merger though; we'll assign labels post-merge based on spatial
        # position (top->bottom). For now mark each polygon with a placeholder.
        for poly in polygons:
            pre_polys.append(poly)
            pre_texts.append("")
    else:
        rec_fn = recognize_mod.REGISTRY[cfg.recognizer]
        for poly in polygons:
            bl = baseline_mod.fit(poly)
            if bl.kind == "arc":
                crop = unwarp_mod.unwarp_arc(img_bgr, poly, bl)
            else:
                crop = _crop_polygon(img_bgr, poly, height_mul=cfg.text_height_factor)
            if crop.size == 0:
                continue
            text, _ = rec_fn(crop[:, :, ::-1])
            clean = "".join(c for c in text if c.isalnum() or c.isspace()).strip().upper()
            pre_texts.append(clean)
            pre_polys.append(poly)

    # Scale arc-residual threshold to image size: 1% of the short edge.
    short_edge = float(min(H, W))
    has_measured = any(mbl is not None for mbl in measured_baselines)
    if has_measured and cfg.text_overrides is not None:
        # Skip merger: each polygon already represents one complete text
        # region and we have a measured baseline for it. Going through the
        # merger would lose the per-polygon baseline mapping.
        merged = list(zip(pre_polys, pre_texts, measured_baselines))
    else:
        merged = merge_mod.maybe_merge(pre_polys, pre_texts,
                                       residual_threshold=max(8.0, 0.01 * short_edge))

    # If text was overridden, assign expected strings by polygon-centroid Y
    # (top-to-bottom).
    if cfg.text_overrides is not None:
        order = sorted(range(len(merged)),
                       key=lambda i: merged[i][0][:, 1].mean())
        new_merged = list(merged)
        for rank, idx in enumerate(order):
            if rank < len(cfg.text_overrides):
                poly_i, _t, bl_i = new_merged[idx]
                new_merged[idx] = (poly_i, cfg.text_overrides[rank], bl_i)
        merged = new_merged

    texts: list[DetectedText] = []
    for poly, joined_text, merged_baseline in merged:
        bl = merged_baseline if merged_baseline is not None else baseline_mod.fit(poly)
        if bl.kind == "arc":
            crop = unwarp_mod.unwarp_arc(img_bgr, poly, bl)
        else:
            crop = _crop_polygon(img_bgr, poly, height_mul=cfg.text_height_factor)
        if crop.size == 0:
            continue
        # If joined_text came out empty (because pre-pass found nothing on a
        # raw curved crop), re-run recognition on the unwarped joined crop --
        # unless we're in text-override mode, in which case any unlabeled
        # polygon is an extra detection (likely a false positive) and we
        # drop it rather than invent OCR for it.
        if not joined_text:
            if cfg.text_overrides is not None:
                continue
            text, conf = rec_fn(crop[:, :, ::-1])
            joined_text = "".join(c for c in text if c.isalnum() or c.isspace()).strip().upper()
            confidence = float(conf)
        else:
            confidence = 0.9
        dt = DetectedText(polygon=poly, text=joined_text, confidence=confidence,
                          baseline=bl, crop=crop)
        texts.append(dt)
    timings["recognize+baseline"] = time.perf_counter() - t0
    out.texts = texts

    # 3. font id
    t0 = time.perf_counter()
    corpus = fontid_mod.discover_corpus(cfg.font_corpus_dir,
                                        min_weight=cfg.corpus_min_weight)
    matches: list[FontMatch] = []
    candidates_per_text: list[list[FontMatch]] = []
    for i, dt in enumerate(texts):
        if not dt.text or not corpus:
            matches.append(FontMatch(family="sans-serif", weight=700))
            candidates_per_text.append([])
            continue
        # Font override -- forces a specific (family, weight, ttf) for this
        # text slot, skipping the search. The optimizer still sweeps size/dx/dy
        # for the override, so the placement is still tuned to the source.
        override = (cfg.font_overrides[i]
                    if cfg.font_overrides and i < len(cfg.font_overrides)
                    else None)
        if override is not None:
            fam, wt, ttf = override
            forced = FontMatch(family=fam, weight=wt, ttf_path=ttf)
            candidates_per_text.append([forced])
            best = forced
        else:
            gray = cv2.cvtColor(dt.crop, cv2.COLOR_BGR2GRAY)
            top_k = max(cfg.top_k_fonts, cfg.refine_top_k if cfg.refine else 0)
            # When refining, skip the per-candidate letter-spacing sweep at the
            # font-ID stage: the optimizer will do a more thorough sweep on the
            # finalists (size, ls, dx, dy, baseline) so we just need a top-K.
            candidates = fontid_mod.match(gray, dt.text, corpus, top_k=top_k,
                                          skip_fine=cfg.refine)
            candidates_per_text.append(candidates)
            best = candidates[0] if candidates else FontMatch(family="sans-serif", weight=700)
        # Re-record a sensible size for assembly.
        # The polygon's y-extent is wrong for both arched (arc sag) and tilted
        # rectangles (rotated, axis-aligned bbox is huge). Use:
        #  - unwarped-strip height for arc baselines
        #  - rotated minAreaRect short side for line baselines
        if dt.baseline and dt.baseline.kind == "arc":
            # Measure actual ink height inside the unwarped strip rather than
            # trusting the strip height (the polygon may have padded the
            # radial extent beyond the glyphs).
            gray_strip = cv2.cvtColor(dt.crop, cv2.COLOR_BGR2GRAY)
            ink = (gray_strip < 128)
            if ink.any():
                row_has_ink = ink.any(axis=1)
                ys = np.where(row_has_ink)[0]
                phys_h = float(ys.max() - ys.min() + 1)
            else:
                phys_h = float(dt.crop.shape[0])
        else:
            rect = cv2.minAreaRect(dt.polygon.astype(np.float32))
            w_rr, h_rr = rect[1]
            phys_h = float(min(w_rr, h_rr))
        best_dict = best.__dict__.copy()
        # font-size ≈ cap-height / 0.72 for typical sans-serif fonts.
        best_dict["size_px"] = phys_h / 0.72
        matches.append(FontMatch(**best_dict))
    timings["fontid"] = time.perf_counter() - t0

    # 3b. local sweep over (size, ls, dy=arc-radius-shift, dx, candidate baseline)
    if cfg.refine:
        t0 = time.perf_counter()
        # Try to locate the badge's center so synthesised arc baselines stay
        # concentric with the rings (especially important for the year text,
        # which is on a wide arc around the same center as CAMP TINKER).
        badge = geom_mod.find_badge_center(img_bgr)
        hint = [(badge[0], badge[1])] if badge else None
        # Per-letter detection serves a different role here: derive a more
        # accurate arc baseline (center + radius + angular span) than the
        # polygon-CircleModel fit. The polygon traces the OUTER edge of
        # the text band, so its arc fit is biased outward; per-letter
        # centroids sit on the actual baseline. We use the per-letter
        # info ONLY to refit the baseline, then keep emitting <textPath>
        # so the SVG remains editable — changing "CAMP TINKER" to
        # "BASE CAMP" should just work without rewriting per-glyph
        # coordinates.
        if badge:
            import math as _math
            for dt in texts:
                anchors = geom_mod.detect_letter_anchors(
                    img_bgr, dt.polygon, dt.text or "",
                    badge_center=(badge[0], badge[1]))
                if anchors:
                    a0 = anchors[0]; an = anchors[-1]
                    cx_a, cy_a, r_a = a0["arc_cx"], a0["arc_cy"], a0["arc_r"]
                    # Sweep ONLY the extreme characters' positions to find
                    # where they truly want to land. Centroid bias for
                    # outer letters (C/R/2/5) is what pulled the text in
                    # too tight; per-letter sweep corrects it. We don't
                    # sweep intermediates — those are derived by font
                    # advances + letter-spacing in assemble.
                    caps = [a["cap_h"] for a in anchors]
                    size = (float(sorted(caps)[len(caps) // 2]) / 0.72) * 0.85
                    ttf = None
                    for over in (cfg.font_overrides or []):
                        if over:
                            ttf = over[2]
                            break
                    # Use RAW bbox centers (not swept) for the arc
                    # endpoints. Swept positions maximize IoU which can
                    # bias INWARD (a rendered glyph that's slightly narrower
                    # than the original lands centered on the mask, not
                    # at the original glyph center). Bbox centers are
                    # unbiased and the per-character sweep in assemble
                    # still corrects the visual placement.
                    t0 = _math.atan2(a0["anchor_y_raw"] - cy_a,
                                     a0["anchor_x_raw"] - cx_a)
                    t1 = _math.atan2(an["anchor_y_raw"] - cy_a,
                                     an["anchor_x_raw"] - cx_a)
                    dt.baseline = Baseline(
                        kind="arc",
                        params=(cx_a, cy_a, r_a, t0, t1),
                        residual=0.0,
                    )
                    dt.letter_anchors = anchors
        if badge:
            out.timings["badge_center"] = 0.0
            out.config["badge_center"] = {"cx": badge[0], "cy": badge[1], "r": badge[2]}
        refined: list[FontMatch] = []
        for dt, fm, cands in zip(texts, matches, candidates_per_text):
            if not cands or not dt.text:
                refined.append(fm)
                continue
            # Use the top-K finalists as the discrete weight/family axis.
            tops = cands[: cfg.refine_top_k]
            # Make sure they all carry the same initial size estimate.
            seeded = [type(c)(**{**c.__dict__, "size_px": fm.size_px}) for c in tops]
            best_fm, params, new_baseline = optimize_mod.sweep(
                dt, seeded, img_bgr,
                try_arc_for_line=cfg.refine_try_arc,
                force_arc=cfg.refine_force_arc,
                arc_centers_hint=hint,
            )
            refined.append(best_fm)
            if new_baseline is not None:
                dt.baseline = new_baseline
        matches = refined
        timings["refine"] = time.perf_counter() - t0
    out.font_matches = matches

    # 4. inpaint
    t0 = time.perf_counter()
    inp_fn = inpaint_mod.REGISTRY[cfg.inpainter]
    mask = inpaint_mod.polygons_to_mask([dt.polygon for dt in texts], (H, W), inflate_px=4)
    residual = inp_fn(img_bgr, mask)
    timings["inpaint"] = time.perf_counter() - t0

    # 4b. paint the rings white in the residual so vtracer doesn't re-trace
    # them as wobbly Beziers alongside our clean <circle> elements; then
    # paint EVERYTHING outside the outer ring white (text and any other
    # non-badge ink). Circles were already detected at step 1b on the
    # original image -- reuse those results, no duplicate detection.
    if found_circles:
        t0 = time.perf_counter()
        residual = circles_mod.paint_white(residual, found_circles, circle_widths)
        outer_cx, outer_cy, outer_r = found_circles[0]
        outer_sw = circle_widths[0]
        yy_idx, xx_idx = np.mgrid[0:H, 0:W]
        outside_outer = (np.hypot(xx_idx - outer_cx, yy_idx - outer_cy)
                          > (outer_r + outer_sw / 2.0 + 2.0))
        if residual.ndim == 3:
            residual[outside_outer] = 255
        else:
            residual[outside_outer] = 255
        timings["paint_rings"] = time.perf_counter() - t0

    # 5. trace -- either vtracer (filled Bezier contours) or skeleton-based
    # (one <line>/<polyline> per centerline). Skeleton is the default for
    # inside-the-rings geometry on this kind of badge: it preserves stroke
    # constraints (fixed width, axis-aligned tree trunks / mountain base /
    # waterfall ticks) that vtracer can't express. The vtracer path is kept
    # as a fallback for designs the skeleton approach doesn't handle.
    t0 = time.perf_counter()
    if cfg.use_skeleton_for_inside and found_circles:
        inner = min(zip(found_circles, circle_widths), key=lambda c: c[0][2])
        (inner_cx, inner_cy, inner_r), inner_sw = inner
        gray = (cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
                if residual.ndim == 3 else residual)
        yy_idx, xx_idx = np.mgrid[0:H, 0:W]
        inside_mask = (np.hypot(xx_idx - inner_cx, yy_idx - inner_cy)
                        < (inner_r - inner_sw / 2.0 - 2.0))
        binary = (gray < 128) & inside_mask
        skel = skeleton_mod.skeletonize_binary(binary)
        skel = skeleton_mod.prune_spurs(
            skel, max_spur_len=max(int(round(inner_sw)), 8))
        skel, _ = skeleton_mod.straighten_segments(skel)
        skel, _ = skeleton_mod.snap_to_ring(
            skel, inner_cx, inner_cy, inner_r - inner_sw / 2.0,
            max_gap=25.0)
        paths = skeleton_mod.skeleton_to_svg_paths(
            skel, stroke_width=inner_sw)
        trace = TraceResult(svg_paths=paths, width=W, height=H)
    else:
        tr_fn = trace_mod.REGISTRY[cfg.tracer]
        trace = tr_fn(residual, **cfg.tracer_kwargs)
    timings["trace"] = time.perf_counter() - t0

    # 5b. Vertical-snap below horizon: clean up the waterfall ticks vtracer
    # leaves tilted a few degrees off vertical. Only kicks in for tall-narrow
    # paths below the detected horizon (mountain baseline) and only when
    # their PCA axis is within tolerance.
    if (cfg.vertical_snap_tolerance_deg > 0 and found_circles
            and trace.svg_paths):
        t0 = time.perf_counter()
        (outer_cx2, outer_cy2, outer_r2) = found_circles[0]
        inner_r2 = found_circles[-1][2] if len(found_circles) > 1 else outer_r2
        gray_residual = (cv2.cvtColor(residual, cv2.COLOR_BGR2GRAY)
                         if residual.ndim == 3 else residual)
        horizon_y = snap_strokes_mod.detect_horizon_y(
            gray_residual, outer_cx2, outer_cy2, inner_r2)
        if horizon_y is not None:
            debug_records: list[dict] = []
            trace = TraceResult(
                svg_paths=snap_strokes_mod.snap_vertical_below_horizon(
                    trace.svg_paths, horizon_y,
                    tolerance_deg=cfg.vertical_snap_tolerance_deg,
                    inner_circle=(outer_cx2, outer_cy2, inner_r2),
                    debug_records=debug_records),
                width=trace.width, height=trace.height,
            )
            out.config["horizon_y"] = float(horizon_y)
            out.config["snap_debug"] = debug_records
            # Save a debug overlay PNG next to the SVG so build_debug_page
            # can surface it. backdrop is the inpainted/painted residual --
            # what vtracer actually saw, so the bbox positions overlay
            # cleanly on the input the snap was reasoning about.
            out_dir = Path(output_path).parent
            out_dir.mkdir(parents=True, exist_ok=True)
            overlay = snap_strokes_mod.render_debug_overlay(
                debug_records,
                img_bgr,
                horizon_y=horizon_y,
            )
            cv2.imwrite(str(out_dir / "snap_debug.png"), overlay)
        timings["snap_vertical"] = time.perf_counter() - t0

    out.trace = trace

    # 6. assemble
    t0 = time.perf_counter()
    assemble(trace, texts, matches, output_path=output_path,
             circles=found_circles, circle_widths=circle_widths,
             img_bgr=img_bgr)
    timings["assemble"] = time.perf_counter() - t0
    timings["total"] = sum(timings.values())
    return out
