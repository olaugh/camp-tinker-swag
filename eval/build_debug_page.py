"""Build a static debug page comparing the original PNG to every run in runs/.

Reads `runs/leaderboard.json` plus per-run `summary.json` and emits a single
self-contained `debug.html` with relative <img> links to the original PNG,
each rendered output, the SVG (inlined via <object>), and the diff map.

Usage:
    .venv/bin/python eval/build_debug_page.py \
        --runs runs --original assets/camp_tinker_synth.png \
        --reference-svg assets/camp_tinker_synth.svg --out debug.html
"""
from __future__ import annotations

import argparse
import datetime
import html
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from lxml import etree

# Make sure src/ is on sys.path so we can import ctswag.skeleton.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctswag import skeleton as skel_mod  # noqa: E402


_FONTS_DIR: Path | None = None


def _rsvg_render(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    """Rasterize svg_path to png_path via resvg, loading TTFs from
    _FONTS_DIR. resvg honors `<textPath>` (rsvg-convert doesn't render it
    at all) and resolves fonts by family name from --use-fonts-dir without
    relying on @font-face. Falls back to cairosvg only if resvg isn't on
    PATH -- but cairosvg ignores both @font-face data URLs and textPath
    layout details, so this is a poor backup.
    """
    cmd = ["resvg",
           "-w", str(width), "-h", str(height)]
    if _FONTS_DIR and _FONTS_DIR.exists():
        cmd += ["--use-fonts-dir", str(_FONTS_DIR.resolve())]
    cmd += [str(svg_path.resolve()), str(png_path)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    import cairosvg
    cairosvg.svg2png(url=str(svg_path.resolve()),
                     write_to=str(png_path),
                     output_width=width, output_height=height)


def fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.0%}"


def fmt_num(x: float | None, digits: int = 3) -> str:
    return "—" if x is None else f"{x:.{digits}f}"


def fmt_fonts(fonts: list) -> str:
    if not fonts:
        return "—"
    return ", ".join(f"{fam} {wt} ({score:.3f})" for fam, wt, score in fonts)


SVG_NS = "http://www.w3.org/2000/svg"


def _strip(svg_path: Path, *, keep: str, out_path: Path) -> None:
    """Write a copy of svg_path with one layer kept.

    keep="text"   -> remove vtracer geometry + rings; leave <text>
    keep="geom"   -> remove every <text>; leave geometry + rings
    keep="rings"  -> remove vtracer geometry + <text>; leave only <circle> rings
    """
    tree = etree.parse(str(svg_path))
    root = tree.getroot()
    text_tag = f"{{{SVG_NS}}}text"
    g_tag = f"{{{SVG_NS}}}g"

    if keep == "geom":
        for t in root.iter(text_tag):
            t.getparent().remove(t)
    elif keep == "text":
        for g in list(root.iter(g_tag)):
            gid = g.get("id") or ""
            if gid in ("geometry", "rings"):
                g.getparent().remove(g)
    elif keep == "rings":
        for t in root.iter(text_tag):
            t.getparent().remove(t)
        for g in list(root.iter(g_tag)):
            gid = g.get("id") or ""
            if gid == "geometry":
                g.getparent().remove(g)
    else:
        raise ValueError(f"keep must be 'geom'|'text'|'rings', got {keep!r}")

    out_path.write_bytes(etree.tostring(root, xml_declaration=True, encoding="UTF-8"))


def _rasterize(svg_path: Path, png_path: Path, width: int, height: int) -> None:
    _rsvg_render(svg_path, png_path, width, height)


def _diff_png(a_path: Path, b_path: Path, out_path: Path) -> None:
    """Write |a - b| as a PNG (a and b resized to match if needed)."""
    a = np.array(Image.open(a_path).convert("RGB"))
    b = np.array(Image.open(b_path).convert("RGB"))
    if a.shape != b.shape:
        b = np.array(Image.fromarray(b).resize((a.shape[1], a.shape[0]),
                                                Image.BILINEAR))
    diff = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)),
                   0, 255).astype(np.uint8)
    Image.fromarray(diff).save(out_path)


def make_region_pngs(run_dir: Path, original_path: Path) -> dict[str, Path]:
    """Split the original PNG into the layers that the detected rings bound:
      inside  = pixels strictly inside the inner ring (line art region)
      outside = pixels strictly outside the outer ring (text region)
    Also write a |original - layer| diff for each.

    Reads ring geometry from summary.json (config.circles[]). If <2 rings are
    detected we skip (no useful boundary to split on)."""
    out: dict[str, Path] = {}
    sp = run_dir / "summary.json"
    if not sp.exists():
        return out
    summary = json.loads(sp.read_text())
    circles = (summary.get("config") or {}).get("circles") or []
    if len(circles) < 2:
        return out

    sorted_by_r = sorted(circles, key=lambda c: -c["r"])
    outer = sorted_by_r[0]
    inner = sorted_by_r[-1]
    cx, cy = float(outer["cx"]), float(outer["cy"])
    outer_outer_edge = outer["r"] + outer["stroke_width"] / 2.0
    inner_inner_edge = inner["r"] - inner["stroke_width"] / 2.0

    orig = np.array(Image.open(original_path).convert("RGB"))
    H, W = orig.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)

    pad = 2.0  # clear the ring's anti-alias edge so it's not in either layer
    inside_mask = dist < (inner_inner_edge - pad)
    outside_mask = dist > (outer_outer_edge + pad)

    inside_layer = np.where(inside_mask[..., None], orig,
                             np.uint8(255)).astype(np.uint8)
    outside_layer = np.where(outside_mask[..., None], orig,
                              np.uint8(255)).astype(np.uint8)

    out["inside"] = run_dir / "inside-layer.png"
    out["outside"] = run_dir / "outside-layer.png"
    out["inside_diff"] = run_dir / "inside-diff.png"
    out["outside_diff"] = run_dir / "outside-diff.png"
    Image.fromarray(inside_layer).save(out["inside"])
    Image.fromarray(outside_layer).save(out["outside"])

    inside_diff = np.clip(np.abs(orig.astype(np.int16)
                                   - inside_layer.astype(np.int16)),
                            0, 255).astype(np.uint8)
    outside_diff = np.clip(np.abs(orig.astype(np.int16)
                                    - outside_layer.astype(np.int16)),
                             0, 255).astype(np.uint8)
    Image.fromarray(inside_diff).save(out["inside_diff"])
    Image.fromarray(outside_diff).save(out["outside_diff"])
    return out


def make_skeleton_pngs(run_dir: Path) -> dict[str, Path]:
    """Skeletonise the inside-the-rings layer, snap endpoints near the inner
    ring to the ring, and re-stroke at the inner ring's stroke width.

    Produces three PNGs next to inside-layer.png:
      skeleton.png         -- 1-pixel centerlines after snapping
      skeleton-cleaned.png -- centerlines dilated back to stroke width
      skeleton-diff.png    -- |inside-layer - skeleton-cleaned|

    Returns {} if the prerequisites (inside-layer, detected rings) are missing.
    """
    out: dict[str, Path] = {}
    inside_path = run_dir / "inside-layer.png"
    sp = run_dir / "summary.json"
    if not inside_path.exists() or not sp.exists():
        return out
    summary = json.loads(sp.read_text())
    circles = (summary.get("config") or {}).get("circles") or []
    if not circles:
        return out
    inner = min(circles, key=lambda c: c["r"])  # innermost ring drives the snap

    inside = np.array(Image.open(inside_path).convert("L"))
    binary = skel_mod.to_binary(inside)
    skel_raw = skel_mod.skeletonize_binary(binary)
    # Prune the medial-axis "Y" forks that appear at every stroke cap --
    # they're not in the source, they're a property of skeletonising a
    # thick stroke. Threshold = roughly the ring stroke width.
    inner_stroke_int = int(round(float(inner["stroke_width"])))
    skel_pruned = skel_mod.prune_spurs(skel_raw,
                                        max_spur_len=max(inner_stroke_int, 8))
    # PCA-fit any straight endpoint-to-endpoint segments (sun rays, waterfall
    # ticks). Mountain/tree skeletons go through junctions and so are skipped.
    skel, n_straightened = skel_mod.straighten_segments(skel_pruned)

    cx = float(inner["cx"])
    cy = float(inner["cy"])
    inner_r = float(inner["r"])
    inner_stroke = float(inner["stroke_width"])
    # Sun rays should reach the inner edge of the inner ring -- that's where
    # the visible ring stroke begins, looking inward from the rays' side.
    target_r = inner_r - inner_stroke / 2.0
    snapped, n_snapped = skel_mod.snap_to_ring(skel, cx, cy, target_r,
                                                max_gap=25.0)

    cleaned = skel_mod.restroke(snapped, inner_stroke)
    cleaned_rgb = np.stack([cleaned] * 3, axis=-1)

    skel_vis = np.where(snapped, np.uint8(0), np.uint8(255))
    skel_rgb = np.stack([skel_vis] * 3, axis=-1)

    out["skeleton"] = run_dir / "skeleton.png"
    out["skeleton_cleaned"] = run_dir / "skeleton-cleaned.png"
    out["skeleton_diff"] = run_dir / "skeleton-diff.png"
    Image.fromarray(skel_rgb).save(out["skeleton"])
    Image.fromarray(cleaned_rgb).save(out["skeleton_cleaned"])

    inside_rgb = np.array(Image.open(inside_path).convert("RGB"))
    diff = np.clip(np.abs(inside_rgb.astype(np.int16)
                           - cleaned_rgb.astype(np.int16)),
                    0, 255).astype(np.uint8)
    Image.fromarray(diff).save(out["skeleton_diff"])
    out["_n_snapped"] = n_snapped  # not a path, but useful for the caption
    return out


def make_layer_pngs(run_dir: Path, render_width: int,
                    original_path: Path) -> dict[str, Path]:
    """Produce per-layer PNGs (geom, text, rings) and a ring-vs-original diff.

    Returns a dict of {key: png_path}. Missing keys mean the layer couldn't
    be rendered (e.g. SVG parse failure, cairo missing).
    """
    svg = run_dir / "output.svg"
    keys = ("geom", "text", "rings")
    out: dict[str, Path] = {k: run_dir / f"{k}-only.png" for k in keys}
    out["rings_diff"] = run_dir / "rings-diff.png"
    if not svg.exists():
        return out
    try:
        orig_w, orig_h = Image.open(original_path).size
        out["geom_svg"] = run_dir / "geom-only.svg"
        out["text_svg"] = run_dir / "text-only.svg"
        out["rings_svg"] = run_dir / "rings-only.svg"
        for keep in keys:
            stripped = run_dir / f"{keep}-only.svg"
            _strip(svg, keep=keep, out_path=stripped)
            if keep == "rings":
                # Render rings at the original image's NATIVE resolution so the
                # rings-diff isn't dominated by bilinear-resize artifacts.
                _rsvg_render(stripped, out[keep], orig_w, orig_h)
            else:
                _rasterize(stripped, out[keep], render_width, render_width)
        # |original - rings_only|: white pixels = original ink NOT covered by
        # the two detected circles (text, sun rays, mountains, trees, year),
        # plus any sub-pixel stroke mismatch where the rings drift from the
        # actual ink. Apples-to-apples now since rings are at native res.
        _diff_png(original_path, out["rings"], out["rings_diff"])
    except Exception as exc:
        print(f"warn: layer split failed for {run_dir.name}: {exc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--original", type=Path, default=Path("assets/camp_tinker_synth.png"))
    ap.add_argument("--reference-svg", type=Path,
                    default=Path("assets/camp_tinker_synth.svg"))
    ap.add_argument("--out", type=Path, default=Path("debug.html"))
    ap.add_argument("--render-width", type=int, default=1024,
                    help="Width to rasterize SVGs at when emitting reference PNGs.")
    ap.add_argument("--fonts", type=Path, default=Path("fonts_curated"),
                    help="Font directory to register with fontconfig for rsvg-convert.")
    ap.add_argument("--top-n", type=int, default=None,
                    help="If set, only the first N runs (after sorting by "
                         "completeness desc, then SSIM desc) are shown.")
    args = ap.parse_args()

    global _FONTS_DIR
    _FONTS_DIR = args.fonts if args.fonts.exists() else None

    leaderboard_path = args.runs / "leaderboard.json"
    leaderboard_by_name: dict[str, dict] = {}
    if leaderboard_path.exists():
        for row in json.loads(leaderboard_path.read_text()):
            leaderboard_by_name[row["name"]] = row

    # Scan every run dir for summary.json; merge with leaderboard rows when
    # both exist. This makes the page survive successive run_combos.py calls
    # that overwrite leaderboard.json with a single new entry.
    run_dirs = sorted(p for p in args.runs.iterdir() if p.is_dir())
    rows: list[dict] = []
    for d in run_dirs:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        lb = leaderboard_by_name.get(d.name, {})
        merged = {
            "name": d.name,
            "wall_s": (s.get("timings") or {}).get("total") or lb.get("wall_s"),
            "ssim": (s.get("metrics_overall") or {}).get("ssim", lb.get("ssim")),
            "l1_255": (s.get("metrics_overall") or {}).get("l1_255", lb.get("l1_255")),
            "ssim_text": (s.get("metrics_text") or {}).get("ssim") if s.get("metrics_text") else lb.get("ssim_text"),
            "l1_255_text": (s.get("metrics_text") or {}).get("l1_255") if s.get("metrics_text") else lb.get("l1_255_text"),
            "completeness": lb.get("completeness"),
            "texts": [t["text"] for t in (s.get("detected_text") or [])],
            "fonts": [(fm["family"], fm["weight"], round(fm.get("score") or 0.0, 4))
                      for fm in (s.get("font_matches") or [])],
            "timings": s.get("timings") or lb.get("timings") or {},
        }
        rows.append(merged)

    # Rank: completeness desc (None last), then SSIM desc.
    rows.sort(key=lambda r: (-(r["completeness"] if r["completeness"] is not None else -1),
                              -(r["ssim"] or 0)))
    if args.top_n is not None:
        rows = rows[: args.top_n]
    leaderboard = rows

    # Project root is debug.html's location; paths are relative to it.
    out_dir = args.out.resolve().parent

    def rel(p: Path) -> str:
        try:
            return str(p.resolve().relative_to(out_dir))
        except ValueError:
            return str(p.resolve())

    rows_html: list[str] = []
    for i, row in enumerate(leaderboard):
        run_dir = args.runs / row["name"]
        svg_path = run_dir / "output.svg"
        render_path = run_dir / "render.png"
        diff_path = run_dir / "diff.png"
        layers = make_layer_pngs(run_dir, args.render_width, args.original)
        regions = make_region_pngs(run_dir, args.original)
        skel_layer = make_skeleton_pngs(run_dir)
        geom_png = layers["geom"]
        text_png = layers["text"]
        rings_png = layers["rings"]
        rings_diff_png = layers["rings_diff"]
        snap_debug_png = run_dir / "snap_debug.png"
        if not snap_debug_png.exists():
            snap_debug_png = None
        geom_svg = layers.get("geom_svg")
        text_svg = layers.get("text_svg")
        rings_svg = layers.get("rings_svg")
        inside_png = regions.get("inside")
        outside_png = regions.get("outside")
        inside_diff_png = regions.get("inside_diff")
        outside_diff_png = regions.get("outside_diff")
        skel_png = skel_layer.get("skeleton")
        skel_cleaned_png = skel_layer.get("skeleton_cleaned")
        skel_diff_png = skel_layer.get("skeleton_diff")
        n_snapped = skel_layer.get("_n_snapped", 0)

        def src_link(path: Path | None, label: str) -> str:
            if path is None or not path.exists():
                return ""
            return f' <a class="src" href="{rel(path)}">[{label}]</a>'
        summary = {}
        sp = run_dir / "summary.json"
        if sp.exists():
            summary = json.loads(sp.read_text())

        texts = ", ".join(html.escape(t) for t in row.get("texts", [])) or "—"
        fonts = fmt_fonts(row.get("fonts", []))
        timings = summary.get("timings") or row.get("timings") or {}
        timings_str = " · ".join(f"{k}={v:.2f}s" for k, v in timings.items()
                                  if k != "total")

        rows_html.append(f"""
        <section class="run" id="run-{i}">
          <header>
            <h2>#{i+1} &nbsp; <code>{html.escape(row['name'])}</code></h2>
            <div class="metrics">
              <span class="m"><b>complete</b> {fmt_pct(row.get('completeness'))}</span>
              <span class="m"><b>ssim</b> {fmt_num(row.get('ssim'))}</span>
              <span class="m"><b>l1/255</b> {fmt_num(row.get('l1_255'), 2)}</span>
              <span class="m"><b>ssim<sub>text</sub></b> {fmt_num(row.get('ssim_text'))}</span>
              <span class="m"><b>l1<sub>text</sub>/255</b> {fmt_num(row.get('l1_255_text'), 2)}</span>
              <span class="m"><b>wall</b> {fmt_num(row.get('wall_s'), 1)}s</span>
            </div>
          </header>
          <div class="layers">
            <figure>
              <figcaption>original{src_link(args.original, "png")}</figcaption>
              <img src="{rel(args.original)}" alt="original">
            </figure>
            <figure>
              <figcaption>render (full SVG){src_link(svg_path, "svg")}</figcaption>
              <img src="{rel(render_path)}" alt="render">
            </figure>
            <figure>
              <figcaption>|orig − render| (overall diff)</figcaption>
              <img src="{rel(diff_path)}" alt="diff">
            </figure>
            <figure>
              <figcaption>rings only (&lt;circle&gt; pair){src_link(rings_svg, "svg")}</figcaption>
              <img src="{rel(rings_png)}" alt="rings only">
            </figure>
            <figure>
              <figcaption>|orig − rings| (what's not on a circle)</figcaption>
              <img src="{rel(rings_diff_png)}" alt="rings diff">
            </figure>
            {(f"<figure><figcaption>inside the inner ring (line-art region){src_link(inside_png, 'png')}</figcaption>"
              f'<img src="{rel(inside_png)}" alt="inside layer"></figure>') if inside_png else ""}
            {("<figure><figcaption>|orig − inside| (everything not inside)</figcaption>"
              f'<img src="{rel(inside_diff_png)}" alt="inside diff"></figure>') if inside_diff_png else ""}
            {(f"<figure><figcaption>outside the outer ring (text region){src_link(outside_png, 'png')}</figcaption>"
              f'<img src="{rel(outside_png)}" alt="outside layer"></figure>') if outside_png else ""}
            {("<figure><figcaption>|orig − outside| (everything not outside)</figcaption>"
              f'<img src="{rel(outside_diff_png)}" alt="outside diff"></figure>') if outside_diff_png else ""}
            {(f"<figure><figcaption>inside skeleton (1-px centerlines, "
              f"{n_snapped} endpoints snapped to inner ring){src_link(skel_png, 'png')}</figcaption>"
              f'<img src="{rel(skel_png)}" alt="skeleton"></figure>') if skel_png else ""}
            {(f"<figure><figcaption>inside cleaned (skeleton re-stroked at ring width){src_link(skel_cleaned_png, 'png')}</figcaption>"
              f'<img src="{rel(skel_cleaned_png)}" alt="skeleton cleaned"></figure>') if skel_cleaned_png else ""}
            {("<figure><figcaption>|inside − cleaned| (closed gaps + AA noise)</figcaption>"
              f'<img src="{rel(skel_diff_png)}" alt="skeleton diff"></figure>') if skel_diff_png else ""}
            <figure>
              <figcaption>geometry only (vtracer + rings){src_link(geom_svg, "svg")}</figcaption>
              <img src="{rel(geom_png)}" alt="geometry only">
            </figure>
            <figure>
              <figcaption>text only (font-ID output){src_link(text_svg, "svg")}</figcaption>
              <img src="{rel(text_png)}" alt="text only">
            </figure>
            {(f"<figure><figcaption>snap debug (green=snapped, orange=eigval, red=angle, gray=skipped){src_link(snap_debug_png, 'png')}</figcaption>"
              f'<img src="{rel(snap_debug_png)}" alt="snap debug"></figure>') if snap_debug_png else ""}
          </div>
          <dl class="kv">
            <dt>texts</dt><dd><code>{texts}</code></dd>
            <dt>fonts (top‑5)</dt><dd><code>{html.escape(fonts)}</code></dd>
            <dt>timings</dt><dd class="small"><code>{html.escape(timings_str)}</code></dd>
            <dt>files</dt><dd class="small">
              <a href="{rel(svg_path)}">output.svg</a> ·
              <a href="{rel(render_path)}">render.png</a> ·
              <a href="{rel(diff_path)}">diff.png</a> ·
              <a href="{rel(sp)}">summary.json</a>
            </dd>
          </dl>
        </section>
        """)

    # Reference (the synth SVG we're trying to reproduce) shown up top.
    # We rasterize the reference SVG via cairosvg (same renderer the eval
    # uses) rather than letting Chrome render it, because the SVG references
    # font-family by name only and Chrome will fall back to Times if the
    # font isn't installed locally. cairosvg resolves through fontconfig.
    ref_block = ""
    if args.reference_svg.exists():
        ref_png = args.out.parent / f".{args.reference_svg.stem}.rendered.png"
        _rsvg_render(args.reference_svg, ref_png,
                     args.render_width, args.render_width)
        ref_block = f"""
        <section class="run" id="reference">
          <header>
            <h2>reference</h2>
            <div class="metrics">
              <span class="m">ground-truth synth badge (target the pipeline is trying to reproduce)</span>
            </div>
          </header>
          <div class="grid2">
            <figure>
              <figcaption>original PNG (input to the pipeline)</figcaption>
              <img src="{rel(args.original)}" alt="original">
            </figure>
            <figure>
              <figcaption>synth SVG, rasterized via cairosvg
                (<a href="{rel(args.reference_svg)}">view raw SVG</a>)</figcaption>
              <img src="{rel(ref_png)}" alt="synth svg rendered">
            </figure>
          </div>
        </section>
        """

    now = datetime.datetime.now().astimezone()
    now_iso = now.isoformat(timespec="seconds")
    now_human = now.strftime("%Y-%m-%d %H:%M:%S %Z")
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ctswag — pipeline debug ({len(leaderboard)} attempts)</title>
  <style>
    :root {{ color-scheme: light dark; }}
    body {{
      font: 14px/1.4 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0; padding: 24px 32px 60px;
      max-width: 1600px;
    }}
    h1 {{ margin: 0 0 4px; }}
    h2 {{ margin: 0; font-size: 17px; }}
    .lead {{ color: #666; margin: 0 0 24px; }}
    .run {{
      border-top: 1px solid #ddd; padding: 24px 0; margin: 0;
    }}
    .run header {{
      display: flex; align-items: baseline; gap: 18px; flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .metrics {{ display: flex; gap: 14px; flex-wrap: wrap; color: #555; }}
    .m b {{ color: #111; }}
    .grid3 {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 8px;
    }}
    .grid5 {{
      display: grid;
      grid-template-columns: repeat(5, 1fr);
      gap: 6px;
    }}
    .layers {{
      display: grid;
      grid-template-columns: repeat(7, 1fr);
      gap: 6px;
    }}
    @media (max-width: 1400px) {{
      .layers {{ grid-template-columns: repeat(4, 1fr); }}
    }}
    .grid2 {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 8px;
    }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 11px; color: #777; padding: 2px 4px; }}
    .src {{ color: #4a84d6; text-decoration: none; opacity: 0.7; }}
    .src:hover {{ text-decoration: underline; opacity: 1; }}
    img {{
      width: 100%; aspect-ratio: 1 / 1; display: block;
      background:
        repeating-conic-gradient(#eee 0 25%, transparent 0 50%) 0 0/16px 16px;
      object-fit: contain;
      border: 1px solid #ddd;
    }}
    dl.kv {{ display: grid; grid-template-columns: 120px 1fr; gap: 4px 16px; margin: 12px 0 0; }}
    dl.kv dt {{ color: #777; }}
    dl.kv dd {{ margin: 0; }}
    dl.kv dd.small {{ font-size: 12px; color: #555; }}
    code {{ font: 12px/1.3 ui-monospace, SFMono-Regular, Menlo, monospace; }}
    a {{ color: #1a4ec9; }}
    @media (prefers-color-scheme: dark) {{
      body {{ background: #111; color: #eee; }}
      .lead, .metrics, figcaption, dl.kv dt, dl.kv dd.small {{ color: #aaa; }}
      .m b {{ color: #fff; }}
      .run {{ border-top-color: #333; }}
      img {{ border-color: #333; background:
        repeating-conic-gradient(#222 0 25%, transparent 0 50%) 0 0/16px 16px; }}
      a {{ color: #7aa6ff; }}
    }}
  </style>
</head>
<body>
  <h1>ctswag — pipeline debug</h1>
  <p class="lead">
    generated <time datetime="{now_iso}">{now_human}</time> ·
    {len(leaderboard)} attempts, ranked by completeness then SSIM (see
    <a href="{rel(leaderboard_path)}">leaderboard.json</a>).
  </p>
  {ref_block}
  {''.join(rows_html)}
</body>
</html>
"""
    args.out.write_text(page)
    print(f"wrote {args.out} ({len(leaderboard)} runs)")


if __name__ == "__main__":
    main()
