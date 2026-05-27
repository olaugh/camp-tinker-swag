"""Bake the badge geometry into a static site for GitHub Pages.

Runs the pipeline once with the default text (CAMP TINKER 2025), strips the
title and year <text> elements out of the resulting SVG, and writes:

    docs/template.svg   — badge geometry only, two empty <g> placeholders
                          where the title and year text will be injected
                          client-side.
    docs/meta.json      — arc baseline params (cx, cy, r, font size, is_top)
                          for the title and year, plus image dimensions.
    docs/nourd-700.woff2 (or .ttf fallback) — the font, served separately
                          rather than re-embedded per-text-change.
    docs/index.html     — the configurator UI.
    docs/app.js         — text placement, SVG mutation, downloads.
    docs/CNAME etc — left to the user's GH Pages setup.

Run:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
        .venv/bin/python scripts/build_static.py
"""
from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib")
os.environ.setdefault("CTSWAG_FONTS_DIR", str(ROOT / "fonts"))

DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
(DOCS / "vendor").mkdir(exist_ok=True)

INPUT_PNG = ROOT / "assets/camp_tinker_2025.png"
NOURD_TTF = ROOT / "fonts/Nourd/nourd-700.ttf"


def bake_template():
    import cv2
    import numpy as np
    from lxml import etree

    from ctswag.pipeline import run, PipelineConfig
    from ctswag.synth import synthesize
    import ctswag.geom as geom_mod

    print("Running pipeline to bake template SVG…")
    img = cv2.imread(str(INPUT_PNG))
    H, W = img.shape[:2]
    badge = synthesize(font_family="Nourd", weight=700, size=max(H, W),
                       title_text="CAMP TINKER", year_text="2025")
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

    svg_path = DOCS / "_pipeline_out.svg"
    cfg = PipelineConfig(
        detector="synth", recognizer="tesseract", inpainter="white",
        tracer="vtracer", potrace_mode="inside_ring_replace",
        potrace_alpha_upper=0.7, potrace_alpha_lower=1.15,
        refine_force_arc=True,
        text_overrides=["CAMP TINKER", "2025"],
        font_overrides=[("Nourd", 700, str(NOURD_TTF))] * 2,
        detector_kwargs={"badge": badge},
        skip_per_letter_sweep=True,  # uniform default placement
    )
    result = run(INPUT_PNG, svg_path, cfg, badge=badge)

    # Pull arc baseline params out of the PipelineResult's text records.
    # texts is in top-to-bottom order: index 0 = title, 1 = year.
    title_dt, year_dt = result.texts[0], result.texts[1]
    title_match, year_match = result.font_matches[0], result.font_matches[1]
    # baseline.params is (cx, cy, r, t0, t1) for arcs.
    t_cx, t_cy, t_r, t_t0, t_t1 = title_dt.baseline.params
    y_cx, y_cy, y_r, y_t0, y_t1 = year_dt.baseline.params

    # is_top sign convention from assemble.py
    def is_top_for(dt):
        if dt.letter_anchors:
            return dt.letter_anchors[0]["y"] < dt.baseline.params[1]
        # Fallback: title is above arc center (cy_a is much larger than text y)
        # because we placed badge year arc with center far below; ditto title
        # which has arc center at badge ring. Use any anchor's polygon midY.
        ys = dt.polygon[:, 1]
        midY = float((ys.min() + ys.max()) / 2)
        return midY < dt.baseline.params[1]

    title_is_top = is_top_for(title_dt)
    year_is_top = is_top_for(year_dt)

    # Cap-height-derived font size: same formula as assemble.py
    def common_size_for(dt):
        if not dt.letter_anchors:
            return None
        caps = [a["cap_h"] for a in dt.letter_anchors]
        return (float(sorted(caps)[len(caps) // 2]) / 0.72) * 0.8255

    title_size = common_size_for(title_dt) or 191.5
    year_size = common_size_for(year_dt) or 183.4

    # Compute the original text's intended arc width (the same target the
    # assemble.py path solves for) AND the letter-spacing that yields it
    # at the default text. The static site uses these as the "default
    # tracking" so CAMP TINKER renders the same as on the badge, and
    # custom text falls back to compressed letter-spacing if it would
    # otherwise exceed 1.1 × this target width.
    from PIL import ImageFont as _IF
    def arc_targets(dt, default_text, size):
        cx_a, cy_a, r_a, t0_b, t1_b = dt.baseline.params
        f = _IF.truetype(str(NOURD_TTF), int(round(size)))
        natural_w = float(f.getlength(default_text))
        w_first = float(f.getlength(default_text[:1])) if default_text else 0
        w_last  = float(f.getlength(default_text[-1:])) if default_text else 0
        center_to_center_arc = abs(t1_b - t0_b) * r_a
        target_total_w = center_to_center_arc + (w_first + w_last) / 2
        n_pairs = max(len(default_text) - 1, 1)
        default_ls = max(0.0, (target_total_w - natural_w) / n_pairs)
        return target_total_w, default_ls

    title_target_w, title_default_ls = arc_targets(title_dt, "CAMP TINKER", title_size)
    year_target_w,  year_default_ls  = arc_targets(year_dt,  "2025",        year_size)

    meta = {
        "width": W, "height": H,
        "title": {
            "cx": t_cx, "cy": t_cy, "r": t_r,
            "is_top": title_is_top,
            "size_px": title_size,
            "default_text": "CAMP TINKER",
            "target_total_w": title_target_w,
            "default_letter_spacing_px": title_default_ls,
        },
        "year": {
            "cx": y_cx, "cy": y_cy, "r": y_r,
            "is_top": year_is_top,
            "size_px": year_size,
            "default_text": "2025",
            "target_total_w": year_target_w,
            "default_letter_spacing_px": year_default_ls,
        },
        "font_family": "Nourd-Bold",
        "font_url": "nourd-700.woff2",
    }

    # Build the template: remove all <text> children from <g id="text">,
    # add two placeholder groups.
    NS = "http://www.w3.org/2000/svg"
    tree = etree.parse(str(svg_path))
    root = tree.getroot()

    # Strip the inline @font-face — we'll serve the font as a separate
    # WOFF2 file linked via a CSS rule the page can apply.
    for style in list(root.iter(f"{{{NS}}}style")):
        style.getparent().remove(style)
    # Strip the (now empty) <defs> too if it has no remaining children.
    for defs in list(root.iter(f"{{{NS}}}defs")):
        if len(defs) == 0:
            defs.getparent().remove(defs)

    # Find <g id="text"> and replace its children with empty placeholders.
    text_g = None
    for g in root.iter(f"{{{NS}}}g"):
        if g.get("id") == "text":
            text_g = g
            break
    if text_g is not None:
        for child in list(text_g):
            text_g.remove(child)
        title_g = etree.SubElement(text_g, f"{{{NS}}}g", id="title-text")
        year_g = etree.SubElement(text_g, f"{{{NS}}}g", id="year-text")

    # Save template.svg
    template_path = DOCS / "template.svg"
    template_path.write_text(etree.tostring(root, xml_declaration=True,
                                              encoding="UTF-8").decode())
    print(f"  wrote {template_path.relative_to(ROOT)}")

    # Save meta.json
    meta_path = DOCS / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  wrote {meta_path.relative_to(ROOT)}")

    # Clean up build artifacts the pipeline drops next to the SVG.
    svg_path.unlink(missing_ok=True)
    for stray in ("snap_debug.png", "out_outlined.pdf", "out_embedded.pdf"):
        (DOCS / stray).unlink(missing_ok=True)

    return meta


def copy_font() -> dict:
    """Ship Nourd as TTF (always — needed for jsPDF embedding and for
    opentype.js parsing) and as WOFF2 (when woff2_compress is available)
    for the CSS @font-face, since WOFF2 is ~3× smaller for first paint.
    Returns {'ttf': '…', 'woff2': '…' | None}."""
    ttf_dst = DOCS / "nourd-700.ttf"
    shutil.copy(NOURD_TTF, ttf_dst)
    print(f"  wrote {ttf_dst.relative_to(ROOT)} "
          f"({ttf_dst.stat().st_size // 1024} KB)")

    woff2_bin = shutil.which("woff2_compress")
    woff2_url = None
    if woff2_bin:
        try:
            subprocess.run([woff2_bin, str(ttf_dst)], check=True,
                           capture_output=True)
            woff2_path = DOCS / "nourd-700.woff2"
            if woff2_path.exists():
                woff2_url = "nourd-700.woff2"
                print(f"  wrote {woff2_path.relative_to(ROOT)} "
                      f"({woff2_path.stat().st_size // 1024} KB)")
        except Exception:
            pass
    else:
        print("  (install `brew install woff2` to also ship .woff2)")
    return {"ttf": "nourd-700.ttf", "woff2": woff2_url}


def write_html_and_js(meta):
    (DOCS / "index.html").write_text(INDEX_HTML)
    (DOCS / "app.js").write_text(APP_JS)
    print(f"  wrote {(DOCS/'index.html').relative_to(ROOT)}")
    print(f"  wrote {(DOCS/'app.js').relative_to(ROOT)}")


# ---------------- HTML + JS ----------------

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Camp Tinker Swag Logo Maker</title>
<style>
  :root { --bg: #1c1c1c; --panel: #262626; --text: #e6e6e6; --muted: #888; --accent: #4eb86b; }
  * { box-sizing: border-box; }
  /* @font-face is registered dynamically from the decrypted TTF
     buffer after the user unlocks — see registerFont() in app.js. */
  body { background: var(--bg); color: var(--text); font: 14px/1.5 -apple-system, "Helvetica Neue", Arial, sans-serif; margin: 0; }
  .app { display: grid; grid-template-columns: 1fr 360px; min-height: 100vh; }
  .preview { padding: 24px; display: flex; align-items: center; justify-content: center; background:
    repeating-conic-gradient(#252525 0 25%, #1c1c1c 0 50%) 50% / 24px 24px; }
  .preview svg, .preview img { max-width: 100%; max-height: calc(100vh - 48px); background: white; border-radius: 6px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); display: block; }
  .sidebar { background: var(--panel); padding: 24px 22px; box-shadow: -1px 0 0 #000; display: flex; flex-direction: column; gap: 18px; }
  h2 { margin: 0; font-size: 18px; font-weight: 600; letter-spacing: 0.02em; }
  .field { display: flex; flex-direction: column; gap: 6px; }
  .field label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); font-weight: 500; }
  .field input[type=text] { background: #2e2e2e; color: var(--text); border: 1px solid #3b3b3b; border-radius: 5px; padding: 8px 10px; font-size: 14px; font-family: inherit; }
  .field input[type=text]:focus { outline: none; border-color: var(--accent); }
  .btn { background: var(--accent); color: #08210f; border: 0; border-radius: 5px; padding: 12px; font-weight: 600; cursor: pointer; font-size: 14px; text-decoration: none; text-align: center; display: block; }
  .btn.secondary { background: #353535; color: var(--text); }
  .btn:disabled { opacity: 0.55; cursor: not-allowed; }
  .row { display: flex; gap: 10px; }
  .row > * { flex: 1; }
  .dl-group { display: flex; flex-direction: column; gap: 6px; }
  .dl-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }
  .status { font-size: 12px; color: var(--muted); min-height: 18px; }
  .status.busy { color: #d4a44e; }
  .status.error { color: #f59292; }
  .warn { background: #3a3320; border-left: 3px solid #d4a44e; padding: 10px 12px; border-radius: 4px; font-size: 12px; color: #e6d9ad; line-height: 1.5; }
  .warn b { color: #ffdf99; }
  .warn ul { margin: 6px 0 0 18px; padding: 0; }
  .footer { color: #555; font-size: 11px; margin-top: auto; }
</style>
</head>
<body>
<div id="lock" style="position:fixed; inset:0; background: var(--bg); display:flex; align-items:center; justify-content:center; z-index:9999;">
  <form id="lock-form" style="background: var(--panel); padding: 28px 28px 24px; border-radius: 8px; min-width: 320px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);">
    <h2 style="margin: 0 0 6px; font-size: 18px;">Camp Tinker Swag Logo Maker</h2>
    <p style="color: var(--muted); margin: 0 0 14px; font-size: 13px;">Enter password to continue.</p>
    <input id="lock-pw" type="password" autocomplete="current-password" autofocus
           style="width: 100%; background: #2e2e2e; color: var(--text); border: 1px solid #3b3b3b; border-radius: 5px; padding: 10px 12px; font-size: 14px; font-family: inherit;">
    <div style="display:flex; gap: 10px; align-items: center; margin-top: 12px;">
      <button type="submit" class="btn" style="flex: 1;">Unlock</button>
    </div>
    <div id="lock-status" style="margin-top: 10px; min-height: 16px; font-size: 12px; color: #f59292;"></div>
  </form>
</div>
<div class="app" style="visibility:hidden">
  <div class="preview" id="preview-wrap">
    <!-- The pre-baked badge SVG is injected here at load. -->
  </div>
  <aside class="sidebar">
    <h2>Camp Tinker Swag Logo Maker</h2>

    <div class="field">
      <label for="title">Title</label>
      <input id="title" type="text" autocomplete="off" spellcheck="false">
    </div>

    <div class="field">
      <label for="year">Year</label>
      <input id="year" type="text" autocomplete="off" spellcheck="false">
    </div>

    <div class="dl-group">
      <div class="dl-label">For print (recommended) ↓</div>
      <div class="row">
        <button id="dl-pdf-out"  class="btn">PDF (outlined)</button>
        <button id="dl-pdf-emb"  class="btn secondary">PDF (text + font)</button>
      </div>
      <div class="dl-label" style="margin-top:8px">Other formats ↓</div>
      <div class="row">
        <button id="dl-svg" class="btn secondary">SVG</button>
        <button id="dl-png" class="btn secondary">PNG</button>
      </div>
    </div>

    <div class="status" id="status">Loading…</div>

    <div class="warn">
      <b>Which should I pick?</b>
      <ul>
        <li>Sending it to a print shop? Use <b>PDF (outlined)</b>.</li>
        <li>Want to edit the text later in Illustrator or Figma? Use <b>PDF (text + font)</b>.</li>
        <li>Putting it on a website? Use <b>SVG</b>.</li>
        <li>Just need an image for email or chat? Use <b>PNG</b>.</li>
      </ul>
      <details style="margin-top:8px">
        <summary style="cursor:pointer; color:#c8b070; font-size:11px; text-transform:uppercase; letter-spacing:0.06em">More detail</summary>
        <ul style="margin-top: 6px">
          <li><b>PDF (outlined)</b> — every letter is converted to a shape, so the file looks identical anywhere it's opened or printed. The text is no longer editable as text.</li>
          <li><b>PDF (text + font)</b> — keeps the text editable and ships the font with the file. Works in Acrobat, Illustrator, etc.</li>
          <li><b>SVG</b> — small, scalable file. Renders correctly in any web browser. Other apps (Preview, Illustrator, Figma) may swap out the font for a generic one unless you install the font on your computer.</li>
          <li><b>PNG</b> — a fixed-size picture. Largest file, but works everywhere.</li>
        </ul>
      </details>
    </div>

  </aside>
</div>

<script src="https://cdn.jsdelivr.net/npm/opentype.js@1.3.4/dist/opentype.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/jspdf@2.5.2/dist/jspdf.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/svg2pdf.js@2.5.0/dist/svg2pdf.umd.min.js"></script>
<script src="app.js"></script>
</body>
</html>
"""

APP_JS = r"""// Static-site configurator: places text on the pre-baked badge SVG and
// handles SVG / PNG / PDF downloads. Everything runs in the browser.

const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  meta: null,
  font: null,         // opentype.js Font
  fontBuffer: null,   // ArrayBuffer for jsPDF embedding
  svg: null,          // the live <svg> element in the preview pane
  titleGroup: null,
  yearGroup: null,
  key: null,          // AES-GCM CryptoKey (after password unlock)
};

// ---- password gate / decryption ----

const KDF_SALT = new TextEncoder().encode("cts-badge-v1");
const KDF_ITERATIONS = 200000;

async function deriveKey(password) {
  const baseKey = await crypto.subtle.importKey(
    "raw", new TextEncoder().encode(password),
    "PBKDF2", false, ["deriveKey"]);
  return crypto.subtle.deriveKey(
    { name: "PBKDF2", salt: KDF_SALT, iterations: KDF_ITERATIONS, hash: "SHA-256" },
    baseKey,
    { name: "AES-GCM", length: 256 },
    false, ["decrypt"]
  );
}

async function decryptBlob(key, buf) {
  const u8 = new Uint8Array(buf);
  const iv = u8.slice(0, 12);
  const ct = u8.slice(12);
  return crypto.subtle.decrypt({ name: "AES-GCM", iv }, key, ct);
}

async function decryptAsText(key, url) {
  const buf = await fetchBuffer(url);
  const plain = await decryptBlob(key, buf);
  return new TextDecoder().decode(plain);
}
async function decryptAsBuffer(key, url) {
  const buf = await fetchBuffer(url);
  return decryptBlob(key, buf);
}
async function decryptAsJson(key, url) {
  return JSON.parse(await decryptAsText(key, url));
}

async function tryUnlock(password) {
  const key = await deriveKey(password);
  // Decrypt verifier — failure throws.
  const buf = await fetchBuffer("verifier.bin");
  let plain;
  try { plain = await decryptBlob(key, buf); }
  catch { throw new Error("Wrong password"); }
  const txt = new TextDecoder().decode(plain);
  if (txt !== "ctswag-ok") throw new Error("Wrong password");
  return key;
}

const titleInput = document.getElementById("title");
const yearInput = document.getElementById("year");
const statusEl = document.getElementById("status");
const previewWrap = document.getElementById("preview-wrap");

function setStatus(msg, cls = "") {
  statusEl.textContent = msg;
  statusEl.className = "status " + cls;
}

async function fetchText(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.text();
}
async function fetchJson(url) { return JSON.parse(await fetchText(url)); }
async function fetchBuffer(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r.arrayBuffer();
}

async function init() {
  setStatus("Loading meta…");
  state.meta = await decryptAsJson(state.key, "meta.json.enc");
  setStatus("Loading template…");
  const tplText = await decryptAsText(state.key, "template.svg.enc");
  setStatus("Loading font…");
  const fontUrl = (state.meta.font_url_ttf || state.meta.font_url) + ".enc";
  state.fontBuffer = await decryptAsBuffer(state.key, fontUrl);
  state.font = opentype.parse(state.fontBuffer);
  // Register the decrypted font with the browser so the live <text>
  // preview renders in real Nourd-Bold (not a system fallback).
  try {
    const face = new FontFace("Nourd-Bold", state.fontBuffer,
                              { weight: "700", style: "normal" });
    await face.load();
    document.fonts.add(face);
  } catch (e) { console.warn("font registration failed:", e); }

  // Inject the template SVG into the preview.
  previewWrap.innerHTML = tplText;
  state.svg = previewWrap.querySelector("svg");
  state.titleGroup = state.svg.querySelector("#title-text");
  state.yearGroup  = state.svg.querySelector("#year-text");
  if (!state.titleGroup || !state.yearGroup) {
    throw new Error("template missing #title-text or #year-text placeholders");
  }

  // Wait for the @font-face Nourd-Bold to actually be ready, so preview
  // shows real Nourd glyphs instead of the FOIT fallback.
  try { await document.fonts.load(`${state.meta.title.size_px}px Nourd-Bold`); }
  catch (e) {}

  titleInput.value = state.meta.title.default_text;
  // Year defaults to the current calendar year on every page load.
  yearInput.value  = String(new Date().getFullYear());
  render();
  setStatus("");

  titleInput.addEventListener("input", render);
  yearInput.addEventListener("input", render);

  document.getElementById("dl-svg").addEventListener("click", () => download_svg());
  document.getElementById("dl-png").addEventListener("click", () => download_png());
  document.getElementById("dl-pdf-out").addEventListener("click", () => download_pdf_outlined());
  document.getElementById("dl-pdf-emb").addEventListener("click", () => download_pdf_embedded());
}

// -------- text placement on arc (port of place_text_on_arc) --------

function placeTextOnArc(text, fontInst, sizePx, arc) {
  // arc = {cx, cy, r, is_top, size_px, target_total_w, default_letter_spacing_px}
  // Returns an array of {char, x, y, rot_deg} for non-space chars.
  if (!text) return [];
  const direction = arc.is_top ? 1 : -1;
  const t_center  = arc.is_top ? -Math.PI / 2 : Math.PI / 2;

  // Per-character natural advances using kerned cumulative measurements.
  const cum = [0];
  for (let i = 0; i < text.length; i++) {
    const w = fontInst.getAdvanceWidth(text.slice(0, i + 1), sizePx,
                                          { kerning: true });
    cum.push(w);
  }
  let advances = [];
  for (let i = 0; i < text.length; i++) advances.push(cum[i + 1] - cum[i]);
  let natural_w = cum[cum.length - 1];
  if (natural_w <= 0) return [];

  // Tracking + sizing rule:
  //   default LS = whatever CAMP TINKER had in the original (baked in).
  //   max total span = 1.1 × target_total_w.
  //   min LS = MIN_LS_RATIO × size_px (so letters never get visually crowded).
  //   1) If natural + (N-1)*default_ls ≤ max → use default_ls.
  //   2) Else if natural + (N-1)*min_ls ≤ max → compress LS to whatever
  //      fits, clamped at MIN_LS (no tighter than that).
  //   3) Else shrink size so natural + (N-1)*min_ls (both scaled) = max,
  //      and keep LS at the scaled min.
  const MIN_LS_RATIO = 0.04;  // 4% of font size — minimum gap between letters
  const n_pairs = Math.max(text.length - 1, 1);
  const default_ls = arc.default_letter_spacing_px || 0;
  const max_total_w = (arc.target_total_w || (natural_w + n_pairs * default_ls)) * 1.10;
  const min_ls = MIN_LS_RATIO * sizePx;
  let letter_spacing_px = default_ls;
  let size_scale = 1.0;
  const natural_with_default = natural_w + n_pairs * default_ls;
  if (natural_with_default > max_total_w) {
    const fit_ls = (max_total_w - natural_w) / n_pairs;
    if (fit_ls >= min_ls) {
      letter_spacing_px = fit_ls;
    } else {
      // Even at the minimum letter-spacing, the text would overflow.
      // Shrink everything proportionally so it fits with min LS preserved.
      // size_scale × (natural_w + n_pairs × min_ls) = max_total_w
      size_scale = max_total_w / (natural_w + n_pairs * min_ls);
      letter_spacing_px = min_ls * size_scale;
    }
  }

  // Apply the size scale to advances + natural width if we had to shrink.
  if (size_scale !== 1.0) {
    advances = advances.map(a => a * size_scale);
    natural_w *= size_scale;
  }
  const effective_size = sizePx * size_scale;

  // Box layout: each char's center is at box_start + advance/2; letter_spacing
  // is added BETWEEN boxes only. Equivalent gap formula:
  //   gap_i,i+1 = (adv_i + adv_{i+1})/2 + letter_spacing_px
  const centers = [];
  let x = 0;
  for (let i = 0; i < text.length; i++) {
    if (i > 0) x += letter_spacing_px;
    centers.push(x + advances[i] / 2);
    x += advances[i];
  }
  const total_w = x;

  const out = [];
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === " ") continue;
    const theta = t_center + direction * (centers[i] - total_w / 2) / arc.r;
    const xPos = arc.cx + arc.r * Math.cos(theta);
    const yPos = arc.cy + arc.r * Math.sin(theta);
    const rot_deg = (theta * 180 / Math.PI) + (arc.is_top ? 90 : -90);
    out.push({ char: ch, x: xPos, y: yPos, rot_deg, size: effective_size });
  }
  return out;
}

function emitTextElements(group, text, arc) {
  // Replace group's children with new <text> elements for this text.
  while (group.firstChild) group.removeChild(group.firstChild);
  const placements = placeTextOnArc(text, state.font, arc.size_px, arc);
  // Use explicit baseline-y instead of dominant-baseline="central":
  // modern browsers respect "central" but svg2pdf does not, so the
  // exported PDF ends up with text shifted up (baseline at the would-be
  // visual center). Computing baseline-y in JS keeps preview and PDF
  // identical, and svg2pdf places text correctly.
  const upem = state.font.unitsPerEm;
  const capH = (state.font.tables.os2 && state.font.tables.os2.sCapHeight) || (upem * 0.72);
  for (const p of placements) {
    const sz = (p.size != null ? p.size : arc.size_px);
    const baselineY = p.y + (capH / upem) * sz / 2;
    const t = document.createElementNS(SVG_NS, "text");
    t.setAttribute("x", p.x.toFixed(2));
    t.setAttribute("y", baselineY.toFixed(2));
    t.setAttribute("text-anchor", "middle");
    t.setAttribute("fill", "black");
    t.setAttribute("font-family", state.meta.font_family);
    t.setAttribute("font-weight", "700");
    t.setAttribute("font-size", sz.toFixed(2));
    // Rotate around the VISUAL center (p.x, p.y), not the baseline point.
    t.setAttribute("transform",
        `rotate(${p.rot_deg.toFixed(2)} ${p.x.toFixed(2)} ${p.y.toFixed(2)})`);
    t.textContent = p.char;
    group.appendChild(t);
  }
}

function render() {
  emitTextElements(state.titleGroup, titleInput.value, state.meta.title);
  emitTextElements(state.yearGroup,  yearInput.value,  state.meta.year);
}

// -------- downloads --------

// Cache the base64-encoded font once; we reuse it for both SVG embedding
// and jsPDF registration.
let _fontB64Cache = null;
function fontBase64() {
  if (_fontB64Cache == null) _fontB64Cache = makeBase64(state.fontBuffer);
  return _fontB64Cache;
}

function svgString() {
  // Serialize the live SVG with @font-face embedding the font as a base64
  // data: URL. Result is fully self-contained — works when the SVG is
  // opened via file:// in any browser that supports SVG @font-face
  // (Chrome, Safari, Firefox).
  const clone = state.svg.cloneNode(true);
  const defs = clone.querySelector("defs") || (() => {
    const d = document.createElementNS(SVG_NS, "defs");
    clone.insertBefore(d, clone.firstChild);
    return d;
  })();
  const style = document.createElementNS(SVG_NS, "style");
  style.setAttribute("type", "text/css");
  style.textContent =
    `@font-face { font-family: 'Nourd-Bold'; font-weight: 700; font-style: normal; ` +
    `src: url(data:font/ttf;base64,${fontBase64()}) format('truetype'); }`;
  defs.appendChild(style);
  return new XMLSerializer().serializeToString(clone);
}

function triggerDownload(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 200);
}

function download_svg() {
  triggerDownload(new Blob([svgString()], { type: "image/svg+xml" }),
                  "camp-tinker.svg");
}

async function download_png() {
  setStatus("Rasterizing PNG…", "busy");
  const svgBlob = new Blob([svgString()], { type: "image/svg+xml" });
  const svgUrl  = URL.createObjectURL(svgBlob);
  const img = new Image();
  img.crossOrigin = "anonymous";
  await new Promise((res, rej) => { img.onload = res; img.onerror = rej; img.src = svgUrl; });
  const c = document.createElement("canvas");
  c.width = state.meta.width;
  c.height = state.meta.height;
  const g = c.getContext("2d");
  g.fillStyle = "white";
  g.fillRect(0, 0, c.width, c.height);
  g.drawImage(img, 0, 0, c.width, c.height);
  await new Promise(res =>
    c.toBlob(b => { triggerDownload(b, "camp-tinker.png"); res(); }, "image/png")
  );
  URL.revokeObjectURL(svgUrl);
  setStatus("PNG downloaded");
}

// PDF: use jsPDF + svg2pdf.js. svg2pdf converts the live SVG to PDF
// using the embedded fonts that jsPDF knows about. We register Nourd
// with jsPDF so the rendered PDF contains real Nourd glyphs.
function makeBase64(arrayBuffer) {
  // Modern browsers: chunked btoa for large arrays
  const bytes = new Uint8Array(arrayBuffer);
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(bin);
}

function newJspdf() {
  const { jsPDF } = window.jspdf;
  const W = state.meta.width, H = state.meta.height;
  // Use points; treat 1 SVG unit = 1 pt for direct geometry mapping.
  const doc = new jsPDF({ unit: "pt", format: [W, H], orientation: W > H ? "l" : "p" });
  // Register Nourd-Bold under EVERY style/weight that svg2pdf might look up.
  // svg2pdf maps SVG font-weight="700" → jsPDF "bold", and many libs strip
  // the "-Bold" suffix from the family name. Cover all cases.
  const b64 = fontBase64();
  const isTTF = state.meta.font_url_ttf.toLowerCase().endsWith(".ttf");
  if (isTTF) {
    doc.addFileToVFS("Nourd-Bold.ttf", b64);
    for (const fam of ["Nourd-Bold", "Nourd"]) {
      for (const style of ["normal", "bold"]) {
        doc.addFont("Nourd-Bold.ttf", fam, style);
      }
    }
  }
  return { doc, hasFont: isTTF };
}

async function download_pdf_embedded() {
  if (!window.jspdf || !window.svg2pdf) {
    setStatus("PDF libs not loaded — check your network", "error");
    return;
  }
  setStatus("Building PDF (embedded)…", "busy");
  const { doc, hasFont } = newJspdf();
  if (!hasFont) {
    setStatus("Embedded-font PDF requires the .ttf font. Re-run build_static.py without WOFF2.", "error");
    return;
  }
  try {
    await window.svg2pdf.svg2pdf(state.svg, doc, {
      x: 0, y: 0, width: state.meta.width, height: state.meta.height,
    });
    doc.save("camp-tinker-embedded.pdf");
    setStatus("PDF downloaded (text editable)");
  } catch (e) {
    console.error(e); setStatus("PDF error: " + e.message, "error");
  }
}

async function download_pdf_outlined() {
  if (!window.jspdf || !window.svg2pdf) {
    setStatus("PDF libs not loaded — check your network", "error");
    return;
  }
  setStatus("Building PDF (outlined)…", "busy");
  // Clone SVG, convert all <text> to filled <path> glyph outlines via
  // opentype.js, then run svg2pdf on the result. The PDF will have no
  // font references at all.
  const clone = state.svg.cloneNode(true);
  const groups = [clone.querySelector("#title-text"),
                  clone.querySelector("#year-text")];
  const defaultSizeFor = (g) => g.id === "title-text" ?
        state.meta.title.size_px : state.meta.year.size_px;
  for (const g of groups) {
    if (!g) continue;
    const newKids = [];
    for (const txt of [...g.querySelectorAll("text")]) {
      const ch = txt.textContent || "";
      if (!ch) continue;
      // Per-letter font-size — may differ from the arc default when the
      // placement algorithm had to shrink the font to fit long text.
      const sz = parseFloat(txt.getAttribute("font-size") || defaultSizeFor(g));
      const cx = parseFloat(txt.getAttribute("x") || "0");
      // y attribute is already the BASELINE (emitTextElements computes it
      // explicitly to keep svg2pdf and browser preview in sync).
      const baselineY = parseFloat(txt.getAttribute("y") || "0");
      const transform = txt.getAttribute("transform") || "";
      // text-anchor=middle → glyph center horizontally at cx
      const otFont = state.font;
      const advance = otFont.getAdvanceWidth(ch, sz);
      const glyphLeftX = cx - advance / 2;
      const path = otFont.getPath(ch, glyphLeftX, baselineY, sz);
      const d = path.toPathData(3);
      const pathEl = document.createElementNS(SVG_NS, "path");
      pathEl.setAttribute("d", d);
      pathEl.setAttribute("fill", txt.getAttribute("fill") || "black");
      if (transform) pathEl.setAttribute("transform", transform);
      newKids.push(pathEl);
    }
    // Replace text children with paths
    while (g.firstChild) g.removeChild(g.firstChild);
    for (const k of newKids) g.appendChild(k);
  }
  const { doc } = newJspdf();
  try {
    await window.svg2pdf.svg2pdf(clone, doc, {
      x: 0, y: 0, width: state.meta.width, height: state.meta.height,
    });
    doc.save("camp-tinker-outlined.pdf");
    setStatus("PDF downloaded (text outlined)");
  } catch (e) {
    console.error(e); setStatus("PDF error: " + e.message, "error");
  }
}

// ---- password-gate boot ----

async function startApp() {
  document.getElementById("lock").style.display = "none";
  document.querySelector(".app").style.visibility = "visible";
  try { await init(); }
  catch (e) { console.error(e); setStatus("Init error: " + e.message, "error"); }
}

const lockForm = document.getElementById("lock-form");
const lockPw = document.getElementById("lock-pw");
const lockStatus = document.getElementById("lock-status");
lockForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  lockStatus.textContent = "Unlocking…";
  lockStatus.style.color = "#c8b070";
  try {
    state.key = await tryUnlock(lockPw.value);
    try { sessionStorage.setItem("ctswag-pw", lockPw.value); } catch {}
    lockStatus.textContent = "";
    await startApp();
  } catch (err) {
    state.key = null;
    lockStatus.style.color = "#f59292";
    lockStatus.textContent = err.message || "Unlock failed";
    lockPw.select();
  }
});

// Auto-unlock from this tab's prior session, if any.
(async () => {
  try {
    const saved = sessionStorage.getItem("ctswag-pw");
    if (!saved) return;
    state.key = await tryUnlock(saved);
    await startApp();
  } catch {}
})();
"""


def encrypt_for_pages(password: str, salt: bytes = b"cts-badge-v1",
                       iterations: int = 200_000):
    """Encrypt the assets with AES-GCM keyed by PBKDF2(password, salt).
    Replaces the cleartext files in docs/ with .enc versions. The
    password is never written to the repo — only the resulting ciphertexts
    and a small verifier blob that the JS uses to recognize the right
    password without trying full asset decryption."""
    import os as _os
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32,
                     salt=salt, iterations=iterations)
    key = kdf.derive(password.encode("utf-8"))
    aesgcm = AESGCM(key)

    def enc_path(p: Path):
        if not p.exists():
            return
        nonce = _os.urandom(12)
        ct = aesgcm.encrypt(nonce, p.read_bytes(), None)
        p.with_suffix(p.suffix + ".enc").write_bytes(nonce + ct)
        p.unlink()
        print(f"  encrypted: {p.name} → {p.name}.enc")

    for name in ["template.svg", "meta.json",
                 "nourd-700.ttf", "nourd-700.woff2"]:
        enc_path(DOCS / name)

    # Verifier: a tiny known plaintext encrypted with the same key. JS
    # decrypts it first; success = right password.
    verifier_plain = b"ctswag-ok"
    nonce = _os.urandom(12)
    (DOCS / "verifier.bin").write_bytes(nonce + aesgcm.encrypt(nonce, verifier_plain, None))
    print(f"  wrote docs/verifier.bin ({iterations} PBKDF2 iters, salt={salt.hex()})")


def main():
    meta = bake_template()
    fonts = copy_font()
    meta["font_url_ttf"]   = fonts["ttf"]
    meta["font_url_woff2"] = fonts["woff2"]
    # Backwards-compat field consumed by app.js's TTF fetch path.
    meta["font_url"] = fonts["ttf"]
    (DOCS / "meta.json").write_text(json.dumps(meta, indent=2))
    write_html_and_js(meta)
    # Encrypt the sensitive assets so the deployed page is gated on a
    # password. The password itself is NEVER committed — it must be
    # supplied via the PAGE_PASSWORD env var at build time. Only the
    # resulting ciphertexts ship in docs/.
    password = os.environ.get("PAGE_PASSWORD")
    if not password:
        raise SystemExit(
            "PAGE_PASSWORD env var is required. Re-run with e.g.\n"
            "    PAGE_PASSWORD=… .venv/bin/python scripts/build_static.py")
    encrypt_for_pages(password)
    print()
    print(f"docs/ ready. Try it locally:")
    print(f"  python -m http.server -d docs 8000")
    print(f"  open http://localhost:8000/")
    print(f"To ship: push docs/ to GitHub and set Pages to deploy from /docs.")


if __name__ == "__main__":
    main()
