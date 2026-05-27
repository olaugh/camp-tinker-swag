"""Local web UI for the Camp Tinker badge configurator.

Run:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        .venv/bin/python scripts/line_editor.py

Open http://localhost:8765/. Type a new title / year / thickness; hit
Re-render to bake the change; download SVG or PNG.

State lives in `runs_editor/`:
    out.svg     — the latest emitted SVG
    render.png  — the latest rasterized preview
    meta.json   — SSIM/L1 metrics from the last run
"""
from __future__ import annotations

import http.server
import json
import os
import sys
import traceback
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
os.environ.setdefault("CTSWAG_FONTS_DIR", str(ROOT / "fonts"))

INPUT_PNG = ROOT / "assets/camp_tinker_2025.png"
NOURD_TTF = ROOT / "fonts/Nourd/nourd-700.ttf"
OUT_DIR = ROOT / "runs_editor"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SVG_PATH = OUT_DIR / "out.svg"
RENDER_PATH = OUT_DIR / "render.png"
META_PATH = OUT_DIR / "meta.json"
PDF_OUTLINED = OUT_DIR / "out_outlined.pdf"
PDF_EMBEDDED = OUT_DIR / "out_embedded.pdf"

# svg_to_pdf is a sibling script — make it importable.
sys.path.insert(0, str(ROOT / "scripts"))
import svg_to_pdf as _svg_to_pdf

DEFAULTS = {"title": "CAMP TINKER", "year": "2025"}


def run_pipeline(title: str, year: str) -> dict:
    import cv2
    import numpy as np
    from PIL import Image
    from skimage.metrics import structural_similarity as ssim

    from ctswag.pipeline import run, PipelineConfig
    from ctswag.synth import synthesize
    import ctswag.geom as geom_mod
    import ctswag.eval as eval_mod

    img = cv2.imread(str(INPUT_PNG))
    H, W = img.shape[:2]
    badge = synthesize(font_family="Nourd", weight=700, size=max(H, W),
                       title_text=title, year_text=year)
    cx, cy, ri_real = geom_mod.find_badge_center(img)
    sb_arr = cv2.imdecode(np.frombuffer(badge.png_bytes, np.uint8),
                          cv2.IMREAD_COLOR)
    sx, sy, sri = geom_mod.find_badge_center(sb_arr)
    scale = ri_real / sri
    badge.arc_center = (cx, cy)
    badge.arc_radius = badge.arc_radius * scale
    badge.title_px = badge.title_px * scale
    yx, yy = badge.year_pos
    badge.year_pos = (cx + (yx - sx) * scale, cy + (yy - sy) * scale)
    badge.year_px = badge.year_px * scale
    badge.width, badge.height = W, H
    badge.year_pos = (881.0, 1793.0)
    badge.year_px = 159.0

    # The per-letter sweep is only correct when the supplied text matches
    # the badge image's ink (i.e. CAMP TINKER on a CAMP TINKER badge).
    # For ANY other text, the sweep snaps each glyph to a mismatched CC
    # mask, scrambling spacing and shifting letters off the arc radius.
    skip_sweep = (title.strip().upper() != "CAMP TINKER"
                  or year.strip() != "2025")
    cfg = PipelineConfig(
        detector="synth",
        recognizer="tesseract",
        inpainter="white",
        tracer="vtracer",
        potrace_mode="inside_ring_replace",
        potrace_alpha_upper=0.7,
        potrace_alpha_lower=1.15,
        refine_force_arc=True,
        text_overrides=[title, year],
        font_overrides=[("Nourd", 700, str(NOURD_TTF))] * 2,
        detector_kwargs={"badge": badge},
        skip_per_letter_sweep=skip_sweep,
    )
    run(INPUT_PNG, SVG_PATH, cfg, badge=badge)

    rend = eval_mod.render_svg(SVG_PATH, width=W, height=H)
    Image.fromarray(rend).save(RENDER_PATH)
    # Generate both PDF variants for download — outlined (text as paths,
    # font-independent) and embedded (Nourd embedded as a real TT font).
    try:
        _svg_to_pdf.svg_to_pdf_outlined(SVG_PATH, PDF_OUTLINED, NOURD_TTF)
    except Exception:
        import traceback as _tb; _tb.print_exc()
    try:
        _svg_to_pdf.svg_to_pdf_embedded(SVG_PATH, PDF_EMBEDDED, NOURD_TTF)
    except Exception:
        import traceback as _tb; _tb.print_exc()
    orig = eval_mod.load_image_rgb(INPUT_PNG)

    go = orig.mean(axis=2) / 255.0
    gr = rend.mean(axis=2) / 255.0
    ssim_full = float(ssim(go, gr, data_range=1.0))
    l1 = float(np.abs(orig.astype(int) - rend.astype(int)).mean())
    meta = {
        "title": title, "year": year,
        "ssim": ssim_full, "l1_255": l1,
        "width": W, "height": H,
    }
    META_PATH.write_text(json.dumps(meta, indent=2))
    return meta


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Camp Tinker</title>
<style>
  :root {
    --bg: #1c1c1c; --panel: #262626; --text: #e6e6e6; --muted: #888;
    --accent: #4eb86b;
  }
  * { box-sizing: border-box; }
  body { background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; margin: 0; }
  .app { display: grid; grid-template-columns: 1fr 360px; min-height: 100vh; }
  .preview { padding: 24px; display: flex; align-items: center; justify-content: center; }
  .preview img { max-width: 100%; max-height: calc(100vh - 48px); background: white; border-radius: 6px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); }
  .sidebar { background: var(--panel); padding: 24px 22px; box-shadow: -1px 0 0 #000; display: flex; flex-direction: column; gap: 18px; }
  h2 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; }
  .field input[type=text] { background: #2e2e2e; color: var(--text); border: 1px solid #3b3b3b; border-radius: 5px; padding: 8px 10px; font-size: 14px; font-family: inherit; }
  .field input[type=text]:focus { outline: none; border-color: var(--accent); }
  .field .range-row { display: flex; align-items: center; gap: 10px; }
  .field .range-row input[type=range] { flex: 1; }
  .field .range-row .val { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; width: 48px; text-align: right; }
  .btn { background: var(--accent); color: #08210f; border: 0; border-radius: 5px; padding: 12px; font-weight: 600; cursor: pointer; font-size: 14px; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn.secondary { background: #353535; color: var(--text); }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  .row a.btn { text-decoration: none; text-align: center; display: block; }
  .status { font-size: 12px; color: var(--muted); min-height: 18px; }
  .status.busy { color: #d4a44e; }
  .status.error { color: #f59292; }
  .metrics { background: #2e2e2e; border-radius: 5px; padding: 10px 12px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); line-height: 1.7; }
  .warn { background: #3a3320; border-left: 3px solid #d4a44e; padding: 10px 12px; border-radius: 4px; font-size: 12px; color: #e6d9ad; line-height: 1.45; }
  .warn code { background: rgba(0,0,0,0.3); padding: 1px 5px; border-radius: 3px; font-size: 11px; }
  .warn strong { color: #ffdf99; }
  .warn ul li { margin: 3px 0; }
  .warn ul li b { color: #ffdf99; }
  .warn ul li em { color: #c8b070; font-style: italic; }
  .dl-group { display: flex; flex-direction: column; gap: 6px; }
  .dl-group .dl-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  .metrics strong { color: var(--text); }
  .footer { color: #555; font-size: 11px; margin-top: auto; }
</style>
</head>
<body>
<div class="app">
  <div class="preview">
    <img id="render" src="/render?t=__T__" alt="rendered badge">
  </div>
  <aside class="sidebar">
    <h2>Camp Tinker badge</h2>

    <div class="field">
      <label for="title">Title</label>
      <input id="title" type="text" value="__TITLE__">
    </div>

    <div class="field">
      <label for="year">Year</label>
      <input id="year" type="text" value="__YEAR__">
    </div>

    <button id="rerun" class="btn">Re-render</button>

    <div class="dl-group">
      <div class="dl-label">For print (recommended) ↓</div>
      <div class="row">
        <a id="dl-pdf-out" class="btn" href="/download/pdf/outlined">PDF (outlined)</a>
        <a id="dl-pdf-emb" class="btn secondary" href="/download/pdf/embedded">PDF (text + font)</a>
      </div>
      <div class="dl-label" style="margin-top:8px">Other formats ↓</div>
      <div class="row">
        <a id="dl-svg" class="btn secondary" href="/download/svg">SVG</a>
        <a id="dl-png" class="btn secondary" href="/download/png">PNG</a>
      </div>
    </div>

    <div class="warn">
      <strong>Format guide:</strong>
      <ul style="margin: 6px 0 0 18px; padding: 0;">
        <li><b>PDF (outlined)</b> — text as filled paths. Renders identically in any viewer or print RIP. <em>Preferred for print shops.</em></li>
        <li><b>PDF (text + font)</b> — text stays editable; Nourd-Bold embedded as a real TrueType font. Acrobat/Illustrator can re-type it.</li>
        <li><b>SVG</b> — embeds Nourd via <code>@font-face</code>. Browsers render correctly; macOS Preview, Illustrator, Figma will substitute a system sans unless Nourd is installed.</li>
        <li><b>PNG</b> — raster fallback, no font dependency.</li>
      </ul>
    </div>

    <div class="status" id="status"></div>

    <div class="footer">localhost:8765 · runs_editor/</div>
  </aside>
</div>
<script>
const titleInput = document.getElementById('title');
const yearInput = document.getElementById('year');
const rerun = document.getElementById('rerun');
const statusEl = document.getElementById('status');
const renderImg = document.getElementById('render');

async function loadMeta() {
  try {
    const r = await fetch('/meta'); const m = await r.json();
    if (!m) return;
    titleInput.value = m.title ?? titleInput.value;
    yearInput.value = m.year ?? yearInput.value;
  } catch (e) {}
}

rerun.addEventListener('click', async () => {
  rerun.disabled = true;
  statusEl.textContent = 'Running pipeline…';
  statusEl.className = 'status busy';
  const t0 = performance.now();
  try {
    const body = JSON.stringify({
      title: titleInput.value, year: yearInput.value,
    });
    const r = await fetch('/rerun', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body });
    const j = await r.json();
    if (!r.ok || j.error) throw new Error(j.error || `rerun failed (${r.status})`);
    const ms = Math.round(performance.now() - t0);
    statusEl.textContent = `OK · ${ms} ms`;
    statusEl.className = 'status';
    renderImg.src = '/render?t=' + Date.now();
  } catch (e) {
    statusEl.textContent = 'Error: ' + e.message;
    statusEl.className = 'status error';
  } finally {
    rerun.disabled = false;
  }
});

loadMeta();
</script>
</body>
</html>
"""


def _load_defaults_for_html() -> dict:
    out = dict(DEFAULTS)
    if META_PATH.exists():
        try:
            m = json.loads(META_PATH.read_text())
            for k in ("title", "year"):
                if k in m:
                    out[k] = m[k]
        except Exception:
            pass
    return out


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def _send(self, code, body, content_type="text/plain", extra_headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra_headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _serve_file(self, path: Path, content_type: str, download_name: str | None = None):
        if not path.exists():
            return self._send(404, "not found")
        headers = {}
        if download_name:
            headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        self._send(200, path.read_bytes(), content_type, extra_headers=headers)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            d = _load_defaults_for_html()
            html = (HTML
                    .replace("__TITLE__", d["title"])
                    .replace("__YEAR__", d["year"])
                    .replace("__T__", str(int(__import__('time').time()))))
            return self._send(200, html, "text/html; charset=utf-8")
        if u.path == "/render":
            if not RENDER_PATH.exists():
                try:
                    run_pipeline(**DEFAULTS)
                except Exception:
                    traceback.print_exc()
            return self._serve_file(RENDER_PATH, "image/png")
        if u.path == "/download/png":
            return self._serve_file(RENDER_PATH, "image/png",
                                    download_name="camp-tinker.png")
        if u.path == "/download/svg":
            return self._serve_file(SVG_PATH, "image/svg+xml",
                                    download_name="camp-tinker.svg")
        if u.path == "/download/pdf/outlined":
            return self._serve_file(PDF_OUTLINED, "application/pdf",
                                    download_name="camp-tinker-outlined.pdf")
        if u.path == "/download/pdf/embedded":
            return self._serve_file(PDF_EMBEDDED, "application/pdf",
                                    download_name="camp-tinker-embedded.pdf")
        if u.path == "/meta":
            data = META_PATH.read_text() if META_PATH.exists() else '{}'
            return self._send(200, data, "application/json")
        return self._send(404, "not found")

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n) if n else b""
        except Exception:
            raw = b""
        if u.path == "/rerun":
            try:
                p = json.loads(raw or b"{}")
                title = str(p.get("title", DEFAULTS["title"])).strip() or DEFAULTS["title"]
                year = str(p.get("year", DEFAULTS["year"])).strip() or DEFAULTS["year"]
                meta = run_pipeline(title, year)
                return self._send(200, json.dumps(meta), "application/json")
            except Exception as e:
                traceback.print_exc()
                return self._send(500, json.dumps({"error": str(e)}),
                                  "application/json")
        return self._send(404, "not found")


def main():
    port = int(os.environ.get("PORT", "8765"))
    addr = ("127.0.0.1", port)
    print(f"Camp Tinker configurator: http://{addr[0]}:{addr[1]}/")
    print(f"           input: {INPUT_PNG}")
    print(f"           state: {OUT_DIR}/")
    if not RENDER_PATH.exists():
        print("First-time render…")
        try:
            run_pipeline(**DEFAULTS)
        except Exception:
            traceback.print_exc()
            print("(continuing — preview will be missing until Re-render)")
    server = http.server.ThreadingHTTPServer(addr, Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main()
