"""Font identification by render-and-diff against a corpus of TTF files.

This is the ranking metric: render each candidate font with the detected
string, align it to the source crop, and compute a pixel-level composite
of L1 + (1 - SSIM). The minimum wins. No NN softmax involved.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import minimize_scalar
from skimage.metrics import structural_similarity as ssim

from .types import FontMatch


# Pull weight from filename like "montserrat-700.ttf"
_WEIGHT_RE = re.compile(r"-(\d{3})\.ttf$", re.IGNORECASE)


def discover_corpus(root: Path | str) -> list[FontMatch]:
    """Walk fonts/ and produce a FontMatch entry per TTF."""
    root = Path(root)
    out: list[FontMatch] = []
    for ttf in sorted(root.rglob("*.ttf")):
        family = ttf.parent.name
        m = _WEIGHT_RE.search(ttf.name)
        weight = int(m.group(1)) if m else 400
        out.append(FontMatch(family=family, weight=weight, ttf_path=str(ttf)))
    return out


def _binarize(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img.mean(axis=2)
    return (img < 128).astype(np.uint8)


def _render_string(text: str, ttf_path: str, *,
                   target_h: int, padding: int = 8,
                   letter_spacing_px: float = 0.0) -> np.ndarray:
    """Render text in BLACK on a WHITE background, sized so glyph height ~= target_h.

    letter_spacing_px adds extra pixels between glyphs after each char.
    """
    probe_size = max(int(target_h * 1.2), 16)
    font = ImageFont.truetype(ttf_path, probe_size)
    ascent, descent = font.getmetrics()
    bbox = font.getbbox(text)
    h_probe = bbox[3] - bbox[1]
    if h_probe < 1:
        h_probe = ascent + descent
    scale = target_h / max(h_probe, 1)
    final_size = max(int(probe_size * scale), 8)
    font = ImageFont.truetype(ttf_path, final_size)

    # Measure each character with the final size; figure total width including spacing.
    char_widths: list[int] = []
    top_min, bot_max = 10**9, -10**9
    for ch in text:
        bb = font.getbbox(ch)
        cw = bb[2] - bb[0]
        char_widths.append(max(cw, 1) if ch != " " else int(final_size * 0.3))
        top_min = min(top_min, bb[1])
        bot_max = max(bot_max, bb[3])
    spacing = max(letter_spacing_px, 0)
    total_w = sum(char_widths) + spacing * max(len(text) - 1, 0)
    canvas_w = int(total_w + 2 * padding)
    canvas_h = int((bot_max - top_min) + 2 * padding)
    img = Image.new("L", (max(canvas_w, 8), max(canvas_h, 8)), color=255)
    draw = ImageDraw.Draw(img)
    x = padding
    for ch, cw in zip(text, char_widths):
        bb = font.getbbox(ch)
        draw.text((x - bb[0], padding - top_min), ch, fill=0, font=font)
        x += cw + spacing
    return np.array(img)


def _crop_text(img_gray: np.ndarray, pad: int = 2, *,
               ink_threshold: int = 128, min_row_ink: int = 3) -> np.ndarray:
    """Tighten a grayscale rendering to its ink bounding box.

    Uses a hard ink threshold (mostly-black) and requires each row/column
    to contain at least `min_row_ink` ink pixels so AA edges from unwarp don't
    leak into the bbox.
    """
    bw = img_gray < ink_threshold
    if not bw.any():
        return img_gray
    row_ink = bw.sum(axis=1)
    col_ink = bw.sum(axis=0)
    valid_rows = np.where(row_ink >= min_row_ink)[0]
    valid_cols = np.where(col_ink >= min_row_ink)[0]
    if valid_rows.size == 0 or valid_cols.size == 0:
        ys, xs = np.where(bw)
        valid_rows = ys
        valid_cols = xs
    y0, y1 = max(valid_rows.min() - pad, 0), min(valid_rows.max() + pad + 1, img_gray.shape[0])
    x0, x1 = max(valid_cols.min() - pad, 0), min(valid_cols.max() + pad + 1, img_gray.shape[1])
    return img_gray[y0:y1, x0:x1]


def _resize_to(img: np.ndarray, h: int, w: int) -> np.ndarray:
    return np.array(Image.fromarray(img).resize((w, h), Image.BILINEAR))


def _score(rendered: np.ndarray, source: np.ndarray) -> tuple[float, float, float]:
    """Returns (composite, l1, ssim_val).

    Scores on the BINARIZED ink mask rather than the grayscale image to avoid
    being dominated by anti-aliasing differences between cairosvg and Pillow.
    A small Gaussian blur is applied before binarising so single-pixel
    misalignments don't cause maximum-distance penalties.
    """
    import cv2
    # Blur slightly to forgive sub-pixel jitter
    r = cv2.GaussianBlur(rendered, (3, 3), 0.7)
    s = cv2.GaussianBlur(source, (3, 3), 0.7)
    rb = (r < 160).astype(np.float32)
    sb = (s < 160).astype(np.float32)
    l1 = float(np.mean(np.abs(rb - sb)))
    try:
        v = float(ssim(rb, sb, data_range=1.0))
    except Exception:
        v = 0.0
    composite = 0.5 * l1 + 0.5 * (1.0 - v)
    return composite, l1, v


def _score_one(args: tuple) -> tuple[FontMatch, float, float, float, float]:
    """Worker for the parallel match() call. Returns (fm, composite, l1, ssim, ls)."""
    fm, text, src_norm, working_h, spacings = args
    src_h, src_w = src_norm.shape
    best_composite, best_l1, best_ssim, best_ls = float("inf"), float("inf"), 0.0, 0.0
    for ls in spacings:
        try:
            rendered = _render_string(text, fm.ttf_path, target_h=working_h,
                                      letter_spacing_px=ls)
            rendered = _crop_text(rendered)
            rendered_norm = _resize_to(rendered, src_h, src_w)
            composite, l1, v = _score(rendered_norm, src_norm)
            width_penalty = abs(rendered.shape[1] - src_w) / max(src_w, 1)
            composite = composite + 0.05 * width_penalty
            if composite < best_composite:
                best_composite, best_l1, best_ssim, best_ls = composite, l1, v, ls
        except Exception:
            continue
    return fm, best_composite, best_l1, best_ssim, best_ls


def match(crop_gray: np.ndarray, text: str,
          corpus: Iterable[FontMatch],
          *, top_k: int = 5,
          working_h: int = 128,
          spacings_px: tuple[float, ...] = (0, 4, 8),
          coarse_k: int = 40,
          workers: int | None = None,
          skip_fine: bool = False) -> list[FontMatch]:
    """Score every (font, weight) on a tight text crop.

    For large corpora a two-stage pass is used:
      1. Coarse pass: render each candidate with letter_spacing=0 only, rank.
      2. Fine pass: for the top `coarse_k` candidates, sweep letter-spacing.

    Workers default to os.cpu_count().
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

    src = _crop_text(crop_gray)
    if src.size == 0:
        return list(corpus)[:top_k]
    src_h, src_w = src.shape
    aspect = src_w / max(src_h, 1)
    norm_w = max(int(working_h * aspect), 16)
    src_norm = _resize_to(src, working_h, norm_w)
    corpus_list = list(corpus)
    if not corpus_list:
        return []

    workers = workers or min(os.cpu_count() or 4, 8)

    def run(items: list[FontMatch], spacings: tuple[float, ...]) -> list[FontMatch]:
        # ThreadPoolExecutor is fine because the per-item work is mostly
        # I/O-bound (file reads) + Pillow rasterization which releases the GIL.
        out: list[FontMatch] = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            args_iter = [(fm, text, src_norm, working_h, spacings) for fm in items]
            for fm, comp, l1, v, ls in ex.map(_score_one, args_iter):
                out.append(replace(fm, score=comp, l1=l1, ssim=v,
                                   size_px=float(src_h),
                                   letter_spacing_em=ls / max(working_h, 1)))
        out.sort(key=lambda fm: fm.score)
        return out

    # Stage 1: coarse pass (single spacing)
    if len(corpus_list) > coarse_k:
        coarse = run(corpus_list, spacings=(0.0,))
        finalists = coarse[:coarse_k]
    else:
        finalists = corpus_list
    if skip_fine:
        return finalists[:top_k]
    # Stage 2: fine pass over finalists with full spacing sweep
    fine = run(finalists, spacings=spacings_px)
    return fine[:top_k]


def render_glyph_sheet(text: str, fm: FontMatch, *, target_h: int = 120) -> np.ndarray:
    """Return an RGB rendering of a font candidate for the HTML report."""
    g = _render_string(text, fm.ttf_path, target_h=target_h)
    return np.stack([g, g, g], axis=-1)
