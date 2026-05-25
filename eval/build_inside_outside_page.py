"""Build an HTML page comparing top-N runs by inside vs outside outer-ring metrics.

Reads runs_real_sweep/leaderboard.json, picks the top-N entries by overall SSIM,
recomputes SSIM + L1 split by inside/outside outer ring, and emits a table with
thumbnails so you can see where each font wins (geometry vs text).

The outer-ring center/radius is hardcoded from the detected circles in this
asset (888, 996, r=663.5); change `RING` if you re-run on a different badge.
"""
from __future__ import annotations

import argparse
import base64
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# Detected outer-ring geometry for assets/camp_tinker_2025.png.
RING = (887.5, 996.0, 663.5)  # (cx, cy, r_outer)


def split_metrics(orig: np.ndarray, rend: np.ndarray) -> dict[str, float]:
    if rend.shape != orig.shape:
        rend = np.array(Image.fromarray(rend).resize(
            orig.shape[1::-1], Image.BILINEAR))
    H, W = orig.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy, r_outer = RING
    inside = np.hypot(xx - cx, yy - cy) < r_outer
    outside = ~inside

    gray_a = orig.mean(axis=2) / 255.0
    gray_b = rend.mean(axis=2) / 255.0
    s_full, ssim_map = ssim(gray_a, gray_b, data_range=1.0, full=True)
    abs_diff = np.abs(gray_a - gray_b)

    # Ink overlap: SSIM/L1 are fooled by "blank where there should be text"
    # because mostly-white images mostly agree. Binarize at 0.5, then compute
    # recall (did we cover the original's black pixels?) and precision (did
    # we avoid adding extra black?) restricted to the outside-ring region
    # where the text lives. A font that renders nothing for "CAMP TINKER"
    # gets recall ≈ 0 even though its SSIM stays around 0.92.
    ink_a = (gray_a < 0.5)
    ink_b = (gray_b < 0.5)
    out_a = ink_a & outside
    out_b = ink_b & outside
    intersection = (out_a & out_b).sum()
    recall = intersection / max(out_a.sum(), 1)
    precision = intersection / max(out_b.sum(), 1)
    union = (out_a | out_b).sum()
    iou = intersection / max(union, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    return {
        "ssim_full": float(s_full),
        "ssim_inside": float(ssim_map[inside].mean()),
        "ssim_outside": float(ssim_map[outside].mean()),
        "l1_full": float(abs_diff.mean()) * 255.0,
        "l1_inside": float(abs_diff[inside].mean()) * 255.0,
        "l1_outside": float(abs_diff[outside].mean()) * 255.0,
        "ink_recall_outside": float(recall),
        "ink_precision_outside": float(precision),
        "ink_iou_outside": float(iou),
        "ink_f1_outside": float(f1),
    }


def to_data_uri(path: Path, *, max_w: int = 240) -> str:
    """Inline a PNG as base64 so the HTML is self-contained. Optionally
    downscale wide images to keep file size reasonable."""
    img = Image.open(path)
    if img.width > max_w:
        new_h = int(img.height * max_w / img.width)
        img = img.resize((max_w, new_h), Image.LANCZOS)
    import io
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs_real_sweep"))
    ap.add_argument("--original", type=Path, default=Path("assets/camp_tinker_2025.png"))
    ap.add_argument("--out", type=Path, default=Path("debug_inside_outside.html"))
    ap.add_argument("--top-n", type=int, default=20)
    ap.add_argument("--rank-source", type=Path, default=None,
                    help="Read the leaderboard from this dir (for ranking) "
                         "but expect renders to live under --runs.")
    args = ap.parse_args()

    lb_path = (args.rank_source or args.runs) / "leaderboard.json"
    lb = json.loads(lb_path.read_text())
    lb = sorted(lb, key=lambda r: -r["ssim"])[: args.top_n]

    orig = np.array(Image.open(args.original).convert("RGB"))
    # Encode the original once and reuse it across every row — viewing the
    # render next to the source beats scrolling back to a header.
    orig_uri = to_data_uri(args.original)

    print(f"Computing metrics for top {len(lb)} runs...")
    rows = []
    for i, r in enumerate(lb, 1):
        run_dir = args.runs / r["name"]
        render_path = run_dir / "render.png"
        # Fallback: if --runs uses the --font-overrides naming convention
        # ("...force-<Fam><wt>-<Fam><wt>") instead of the per-font-title
        # naming ("...font<N>-<Fam><wt>"), translate using the leaderboard's
        # captured font tuple.
        if not render_path.exists() and r.get("fonts"):
            fam, wt, _ = r["fonts"][0]
            alt = (args.runs / f"camp_geom_tesseract_telea_vtracer_"
                                f"force-{fam}{wt}-{fam}{wt}")
            if (alt / "render.png").exists():
                run_dir = alt
                render_path = run_dir / "render.png"
        diff_path = run_dir / "diff.png"
        if not render_path.exists():
            print(f"  [{i}] MISSING render: {r['name']}")
            continue
        rend = np.array(Image.open(render_path).convert("RGB"))
        m = split_metrics(orig, rend)
        # Pretty font label
        font_name = r["name"].split("font")[-1].split("-", 1)[-1]
        rows.append({
            "rank": i,
            "font": font_name,
            "render": to_data_uri(render_path),
            "diff": to_data_uri(diff_path) if diff_path.exists() else "",
            **m,
        })
        print(f"  [{i}] {font_name:36s} inside_ssim={m['ssim_inside']:.4f}  "
              f"inside_l1={m['l1_inside']:.2f}  "
              f"ink_f1={m['ink_f1_outside']:.3f}  recall={m['ink_recall_outside']:.3f}")

    # Sort by outside-ring ink F1 (combines recall + precision). That's the
    # metric SSIM/L1 can't see: a font that left the text blank gets F1≈0
    # even though its SSIM is ≈0.92.
    rows.sort(key=lambda r: -r["ink_f1_outside"])
    # Renumber rank by F1 position so the rank column matches the page's
    # actual ordering (not the original sweep's full-SSIM ranking).
    for new_i, r in enumerate(rows, 1):
        r["rank"] = new_i

    rows_html = []
    for r in rows:
        # Color-code F1: green ≥0.6, yellow 0.4-0.6, red <0.4
        f1 = r["ink_f1_outside"]
        f1_cls = "good" if f1 >= 0.6 else ("warn" if f1 >= 0.4 else "bad")
        rows_html.append(f"""
        <tr>
          <td class="rank">#{r['rank']}</td>
          <td class="font">{r['font']}</td>
          <td><img src="{orig_uri}" alt="original"/></td>
          <td><img src="{r['render']}" alt="render"/></td>
          <td><img src="{r['diff']}" alt="diff"/></td>
          <td class="num">{r['ssim_full']:.4f}</td>
          <td class="num good">{r['ssim_inside']:.4f}</td>
          <td class="num dim">{r['ssim_outside']:.4f}</td>
          <td class="num">{r['l1_full']:.2f}</td>
          <td class="num good">{r['l1_inside']:.2f}</td>
          <td class="num dim">{r['l1_outside']:.2f}</td>
          <td class="num {f1_cls}">{r['ink_f1_outside']:.3f}</td>
          <td class="num dim">{r['ink_recall_outside']:.3f}</td>
          <td class="num dim">{r['ink_precision_outside']:.3f}</td>
        </tr>""")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Inside/Outside Outer-Ring Metrics — Top {args.top_n}</title>
<style>
  body {{ font-family: -apple-system, sans-serif; background: #111; color: #eee; padding: 20px; }}
  h1 {{ margin-top: 0; }}
  .meta {{ color: #888; font-size: 13px; margin-bottom: 18px; }}
  table {{ border-collapse: collapse; }}
  th, td {{ padding: 6px 10px; vertical-align: middle; border-bottom: 1px solid #333; }}
  th {{ position: sticky; top: 0; background: #1a1a1a; text-align: left; font-weight: 600; font-size: 13px; }}
  td.rank {{ font-family: monospace; color: #888; }}
  td.font {{ font-family: monospace; max-width: 200px; }}
  td.num {{ font-family: monospace; text-align: right; }}
  td.num.good {{ color: #6fc06f; font-weight: 600; }}
  td.num.warn {{ color: #d4b04f; font-weight: 600; }}
  td.num.bad  {{ color: #c06f6f; font-weight: 600; }}
  td.num.dim  {{ color: #777; }}
  img {{ display: block; width: 180px; image-rendering: pixelated; background: #fff; }}
  caption {{ caption-side: top; text-align: left; padding: 8px 0; color: #aaa; }}
</style></head><body>
<h1>Top {args.top_n} by Inside-Outer-Ring SSIM</h1>
<div class="meta">
  Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} — outer ring at (cx={RING[0]}, cy={RING[1]}, r={RING[2]}).
  Rows sorted by <b>SSIM inside</b> (descending). Original ranking (column #) was by full-image SSIM.
</div>
<table>
  <thead><tr>
    <th>Rank</th><th>Font</th><th>Original</th><th>Render</th><th>Diff</th>
    <th colspan="3" style="text-align:center;">SSIM</th>
    <th colspan="3" style="text-align:center;">L1 / 255</th>
    <th colspan="3" style="text-align:center;">Text Ink (outside ring)</th>
  </tr><tr>
    <th></th><th></th><th></th><th></th><th></th>
    <th>full</th><th>inside</th><th>outside</th>
    <th>full</th><th>inside</th><th>outside</th>
    <th>F1</th><th>recall</th><th>precision</th>
  </tr></thead>
  <tbody>{"".join(rows_html)}</tbody>
</table>
</body></html>
"""
    args.out.write_text(html)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
