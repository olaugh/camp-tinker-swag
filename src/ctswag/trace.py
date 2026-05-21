"""Raster→SVG vectorization adapters."""
from __future__ import annotations

import importlib.util
import re
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from .types import TraceResult


Tracer = Callable[[np.ndarray], TraceResult]


_PATH_TAG_RE = re.compile(r'<path\b[^/>]*/?>', re.IGNORECASE)
_D_RE = re.compile(r'\bd="([^"]+)"', re.IGNORECASE)
_TRANSFORM_RE = re.compile(r'\btransform="([^"]+)"', re.IGNORECASE)
_FILL_RE = re.compile(r'\bfill="([^"]+)"', re.IGNORECASE)


def _extract_paths(svg_text: str) -> list[str]:
    """Return inline-ready <path .../> strings preserving d, transform, fill."""
    out: list[str] = []
    for tag in _PATH_TAG_RE.findall(svg_text):
        d_m = _D_RE.search(tag)
        if not d_m:
            continue
        d = d_m.group(1)
        attrs = [f'd="{d}"']
        tm = _TRANSFORM_RE.search(tag)
        if tm:
            attrs.append(f'transform="{tm.group(1)}"')
        fm = _FILL_RE.search(tag)
        if fm:
            attrs.append(f'fill="{fm.group(1)}"')
        out.append(f"<path {' '.join(attrs)}/>")
    return out


def trace_vtracer(img_bgr: np.ndarray, *,
                  filter_speckle: int = 4,
                  corner_threshold: int = 60,
                  length_threshold: float = 4.0,
                  splice_threshold: int = 45,
                  path_precision: int = 2,
                  colormode: str = "binary") -> TraceResult:
    import vtracer
    H, W = img_bgr.shape[:2]
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / "in.png"
        out_path = Path(td) / "out.svg"
        Image.fromarray(img_bgr[:, :, ::-1]).save(in_path)
        vtracer.convert_image_to_svg_py(
            str(in_path), str(out_path),
            colormode=colormode, hierarchical="stacked", mode="spline",
            filter_speckle=filter_speckle,
            corner_threshold=corner_threshold,
            length_threshold=length_threshold,
            splice_threshold=splice_threshold,
            path_precision=path_precision,
        )
        svg_text = out_path.read_text()
    return TraceResult(svg_paths=_extract_paths(svg_text), width=W, height=H)


def trace_potrace(img_bgr: np.ndarray, *, turdsize: int = 4, alphamax: float = 1.0) -> TraceResult:
    """Pure-python potrace port (tatarize/potrace, GPL)."""
    try:
        import potracer as potrace  # the pure-python port
    except ImportError:
        import potrace  # fallback
    H, W = img_bgr.shape[:2]
    gray = img_bgr.mean(axis=2)
    bm = (gray < 128).astype(np.uint32)
    bmp = potrace.Bitmap(bm)
    path = bmp.trace(turdsize=turdsize, alphamax=alphamax)
    paths = []
    for curve in path:
        d = []
        start_x, start_y = curve.start_point
        d.append(f"M {start_x:.2f} {start_y:.2f}")
        for seg in curve.segments:
            if seg.is_corner:
                cx, cy = seg.c
                ex, ey = seg.end_point
                d.append(f"L {cx:.2f} {cy:.2f} L {ex:.2f} {ey:.2f}")
            else:
                c1x, c1y = seg.c1
                c2x, c2y = seg.c2
                ex, ey = seg.end_point
                d.append(f"C {c1x:.2f} {c1y:.2f} {c2x:.2f} {c2y:.2f} {ex:.2f} {ey:.2f}")
        d.append("Z")
        paths.append(f'<path d="{" ".join(d)}" fill="black"/>')
    return TraceResult(svg_paths=paths, width=W, height=H)


REGISTRY: dict[str, Tracer] = {
    "vtracer": trace_vtracer,
    "potrace": trace_potrace,
}


def have_tracer(name: str) -> bool:
    if name == "vtracer":
        return importlib.util.find_spec("vtracer") is not None
    if name == "potrace":
        return importlib.util.find_spec("potracer") is not None or importlib.util.find_spec("potrace") is not None
    return False
