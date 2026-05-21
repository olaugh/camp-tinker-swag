"""Evaluation helpers: render an SVG, diff against the original, log results."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import cairosvg
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim


def render_svg(svg_path: Path | str, *, width: int, height: int) -> np.ndarray:
    """Rasterize an SVG via cairosvg. Returns RGB ndarray."""
    svg_path = Path(svg_path)
    png_bytes = cairosvg.svg2png(url=str(svg_path), output_width=width, output_height=height)
    img = Image.open(__import__("io").BytesIO(png_bytes)).convert("RGB")
    return np.array(img)


def load_image_rgb(path: Path | str) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("RGB"))


def metrics(a: np.ndarray, b: np.ndarray,
            *, mask: np.ndarray | None = None) -> dict[str, float]:
    """Pairwise metrics between two RGB arrays. mask is optional uint8 mask
    restricting to a region of interest."""
    if a.shape != b.shape:
        H, W = a.shape[:2]
        b = np.array(Image.fromarray(b).resize((W, H), Image.BILINEAR))
    af = a.astype(np.float32) / 255.0
    bf = b.astype(np.float32) / 255.0

    if mask is not None:
        m = (mask > 0).astype(np.float32)
        if m.ndim == 2:
            m3 = m[..., None]
        else:
            m3 = m
        denom = max(float(m.sum()), 1.0)
        l1 = float(np.sum(np.abs(af - bf) * m3) / (denom * af.shape[-1]))
        l2 = float(np.sqrt(np.sum(((af - bf) ** 2) * m3) / (denom * af.shape[-1])))
    else:
        l1 = float(np.mean(np.abs(af - bf)))
        l2 = float(np.sqrt(np.mean((af - bf) ** 2)))

    gray_a = af.mean(axis=2)
    gray_b = bf.mean(axis=2)
    try:
        s = float(ssim(gray_a, gray_b, data_range=1.0))
    except Exception:
        s = 0.0

    return {"ssim": s, "l1": l1, "l1_255": l1 * 255.0, "rmse": l2}


def save_run(out_dir: Path | str, name: str,
             *, original: np.ndarray, rendered: np.ndarray,
             metrics_overall: dict[str, float],
             metrics_text: dict[str, float] | None,
             timings: dict[str, float],
             config: dict[str, Any],
             texts: list[Any],
             font_matches: list[Any]) -> Path:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = out_dir / name
    run_dir.mkdir(exist_ok=True)
    Image.fromarray(rendered).save(run_dir / "render.png")
    # diff
    diff = np.clip(np.abs(original.astype(np.int16) - rendered.astype(np.int16)), 0, 255).astype(np.uint8)
    Image.fromarray(diff).save(run_dir / "diff.png")
    # serializable summary
    summary: dict[str, Any] = {
        "name": name,
        "config": config,
        "timings": timings,
        "metrics_overall": metrics_overall,
        "metrics_text": metrics_text,
        "detected_text": [
            {"text": t.text, "conf": t.confidence,
             "baseline": (t.baseline.kind if t.baseline else None)}
            for t in texts
        ],
        "font_matches": [
            {"family": fm.family, "weight": fm.weight, "score": fm.score,
             "ssim": fm.ssim, "l1": fm.l1, "ttf": fm.ttf_path}
            for fm in font_matches
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return run_dir
