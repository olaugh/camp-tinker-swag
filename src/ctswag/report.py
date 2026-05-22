"""Build an HTML report from a runs/ directory full of combo outputs."""
from __future__ import annotations

import base64
import json
from pathlib import Path


HTML_HEAD = """<!doctype html>
<html><head><meta charset="utf-8"><title>ctswag report</title>
<style>
body { font-family: -apple-system, system-ui, sans-serif; margin: 24px; background: #fafafa; color: #222; }
h1 { margin-top: 0; }
table { border-collapse: collapse; margin: 12px 0; }
th, td { padding: 6px 10px; border-bottom: 1px solid #ddd; vertical-align: top; }
th { background: #eee; text-align: left; }
.row { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 24px; padding: 12px; background: white; border: 1px solid #ddd; border-radius: 6px; }
.row h3 { margin-top: 0; }
.row img { max-width: 100%; height: auto; border: 1px solid #eee; }
.score-good { color: #0a7d28; font-weight: bold; }
.score-mid  { color: #b58400; }
.score-bad  { color: #a02020; }
.code { font-family: ui-monospace, Menlo, monospace; font-size: 12px; }
.legend { font-size: 12px; color: #555; margin-bottom: 12px; }
</style></head>
<body>
"""

HTML_TAIL = "</body></html>"


def _b64_img(path: Path) -> str:
    if not path.exists():
        return ""
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _score_class(ssim: float) -> str:
    if ssim >= 0.95: return "score-good"
    if ssim >= 0.85: return "score-mid"
    return "score-bad"


def build(runs_dir: Path | str, *, original_path: Path | str | None = None,
          out_path: Path | str | None = None) -> Path:
    runs_dir = Path(runs_dir)
    leaderboard_path = runs_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(leaderboard_path)
    rows = json.loads(leaderboard_path.read_text())
    out_path = Path(out_path) if out_path else runs_dir / "report.html"
    lines = [HTML_HEAD, "<h1>ctswag pipeline evaluation</h1>"]
    if original_path:
        lines.append(f'<p><b>Source:</b> <code>{original_path}</code></p>')
        lines.append(f'<img src="{_b64_img(Path(original_path))}" style="max-width:320px;border:1px solid #ddd"/>')
    lines.append('<p class="legend">SSIM is structural similarity (1.0=identical). L1 is mean abs pixel diff. COMP is the fraction of expected text strings found.</p>')

    # Leaderboard table
    lines.append("<h2>Leaderboard</h2><table>")
    lines.append("<tr><th>#</th><th>name</th><th>completeness</th><th>SSIM</th><th>L1/255</th><th>SSIM (text)</th><th>L1/255 (text)</th><th>wall (s)</th><th>found</th><th>fonts</th></tr>")
    for i, row in enumerate(rows, 1):
        comp = row.get("completeness")
        comp_s = f"{comp:.0%}" if comp is not None else "n/a"
        ssim_class = _score_class(row["ssim"])
        ssim_t = row.get("ssim_text") or 0.0
        l1_t = row.get("l1_255_text") or 0.0
        fonts_s = ", ".join(f"{f[0]}/{f[1]}" for f in row.get("fonts", []))
        lines.append(
            f"<tr><td>{i}</td><td><a href='#{row['name']}'>{row['name']}</a></td>"
            f"<td>{comp_s}</td>"
            f"<td class='{ssim_class}'>{row['ssim']:.3f}</td>"
            f"<td>{row['l1_255']:.2f}</td>"
            f"<td>{ssim_t:.3f}</td><td>{l1_t:.2f}</td>"
            f"<td>{row['wall_s']:.1f}</td>"
            f"<td class='code'>{row.get('texts', [])}</td>"
            f"<td class='code'>{fonts_s}</td></tr>"
        )
    lines.append("</table>")

    # Per-row details
    lines.append("<h2>Details</h2>")
    for row in rows:
        rd = runs_dir / row["name"]
        render = rd / "render.png"
        diff = rd / "diff.png"
        ssim_class = _score_class(row["ssim"])
        lines.append(f'<div class="row" id="{row["name"]}">')
        lines.append(f'<div><h3>{row["name"]}</h3>'
                     f'<p>SSIM <span class="{ssim_class}">{row["ssim"]:.3f}</span><br>'
                     f'L1 {row["l1_255"]:.2f}/255<br>'
                     f'wall {row["wall_s"]:.1f}s</p>'
                     f'<p>texts: <span class="code">{row.get("texts", [])}</span></p>'
                     f'<p>fonts: <span class="code">{row.get("fonts", [])}</span></p>'
                     f'<p>timings (s): <span class="code">{row.get("timings", {})}</span></p>'
                     '</div>')
        lines.append(f'<div><h4>render</h4><img src="{_b64_img(render)}"/></div>')
        lines.append(f'<div><h4>diff vs source</h4><img src="{_b64_img(diff)}"/></div>')
        lines.append('</div>')

    lines.append(HTML_TAIL)
    out_path.write_text("\n".join(lines))
    return out_path


if __name__ == "__main__":
    import argparse, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs", type=Path)
    p.add_argument("--original", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    out = build(args.runs, original_path=args.original, out_path=args.out)
    print(f"wrote {out}")
