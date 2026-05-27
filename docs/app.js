// Static-site configurator: places text on the pre-baked badge SVG and
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
