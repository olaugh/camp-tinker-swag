"""Convert the configurator's SVG output to PDF in two flavors:

    outlined  — all text replaced with filled glyph outlines (no font
                dependency at all; renders identically in any PDF viewer
                or print RIP).
    embedded  — text remains text, with the TTF embedded as a real PDF
                font (renders correctly in any conformant PDF viewer,
                stays editable as text in Acrobat/Illustrator).

The two functions accept the same args and produce a PDF at `pdf_path`.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable

# SVG path d="..." tokens
_PATH_TOKEN_RE = re.compile(r'([MLCZAHVQTSmlczahvqts])|([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)')
_TRANSLATE_RE = re.compile(r'translate\(\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*)\s*\)')
_ROTATE_RE = re.compile(r'rotate\(\s*(-?\d+\.?\d*)(?:\s*[,\s]\s*(-?\d+\.?\d*)\s*[,\s]\s*(-?\d+\.?\d*))?\s*\)')

SVG_NS = "http://www.w3.org/2000/svg"


def _parse_d(d: str) -> list[tuple[str, list[float]]]:
    """Parse an SVG path d-string. Returns [(command, [numbers])].
    Repeated coordinate groups under one command are expanded as separate
    cmd entries (so 'M 10 20 30 40' yields M(10,20) then L(30,40))."""
    tokens = []
    for m in _PATH_TOKEN_RE.finditer(d):
        if m.group(1):
            tokens.append(('cmd', m.group(1)))
        else:
            tokens.append(('num', float(m.group(2))))
    out: list[tuple[str, list[float]]] = []
    i = 0
    n_args = {'M': 2, 'L': 2, 'C': 6, 'Z': 0, 'H': 1, 'V': 1, 'Q': 4, 'S': 4, 'T': 2, 'A': 7}
    while i < len(tokens):
        kind, val = tokens[i]
        if kind != 'cmd':
            i += 1
            continue
        cmd = val
        u = cmd.upper()
        nargs = n_args.get(u, 0)
        i += 1
        if nargs == 0:
            out.append((cmd, []))
            continue
        first = True
        while True:
            args = []
            for _ in range(nargs):
                if i >= len(tokens) or tokens[i][0] != 'num':
                    break
                args.append(tokens[i][1])
                i += 1
            if len(args) < nargs:
                break
            out.append((cmd if first else ('L' if u == 'M' else cmd), args))
            first = False
            if i >= len(tokens) or tokens[i][0] != 'num':
                break
    return out


def _parse_transform(s: str) -> tuple[float, float, float, float, float]:
    """Parse SVG transform → (tx, ty, angle_deg, rot_cx, rot_cy)."""
    tx = ty = rot = rcx = rcy = 0.0
    tm = _TRANSLATE_RE.search(s)
    if tm:
        tx, ty = float(tm.group(1)), float(tm.group(2))
    rm = _ROTATE_RE.search(s)
    if rm:
        rot = float(rm.group(1))
        if rm.group(2):
            rcx, rcy = float(rm.group(2)), float(rm.group(3))
    return tx, ty, rot, rcx, rcy


# ---------- 1) OUTLINED ----------

def outline_text_in_svg(svg_text: str, ttf_path: str) -> str:
    """Walk an SVG and replace every <text> (per-letter or textPath) with
    a <g> of filled glyph <path>s. Result has no font dependency."""
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    from lxml import etree

    ttf = TTFont(ttf_path)
    glyph_set = ttf.getGlyphSet()
    cmap = ttf.getBestCmap()
    upem = ttf["head"].unitsPerEm
    cap_h = ttf["OS/2"].sCapHeight or upem * 0.72
    hmtx = ttf["hmtx"]

    def glyph_outline(c: str) -> tuple[str, float] | None:
        """Returns (d-string, advance_units) or None for missing glyphs."""
        gn = cmap.get(ord(c))
        if not gn:
            return None
        glyph = glyph_set[gn]
        pen = SVGPathPen(glyph_set)
        glyph.draw(pen)
        adv = hmtx[gn][0]
        return pen.getCommands(), adv

    parser = etree.XMLParser(remove_blank_text=False, huge_tree=True)
    root = etree.fromstring(svg_text.encode("utf-8"), parser)

    # Index arc defs so we can resolve textPath references.
    arcs: dict[str, dict] = {}
    for p in root.iter(f"{{{SVG_NS}}}path"):
        pid = p.get("id", "")
        if pid.startswith("arc"):
            d = p.get("d", "")
            arc_info = _parse_arc_path(d)
            if arc_info:
                arcs[pid] = arc_info

    # Iterate over a snapshot list so we can mutate during walk.
    for text_el in list(root.iter(f"{{{SVG_NS}}}text")):
        textpath = text_el.find(f"{{{SVG_NS}}}textPath")
        if textpath is not None:
            new_g = _outline_textpath(text_el, textpath, arcs, glyph_outline,
                                     upem, cap_h, ttf_path)
        else:
            new_g = _outline_per_letter(text_el, glyph_outline, upem, cap_h)
        if new_g is None:
            continue
        parent = text_el.getparent()
        idx = parent.index(text_el)
        parent.remove(text_el)
        parent.insert(idx, new_g)

    # Drop the <defs><style>@font-face… block — no longer needed.
    for style in list(root.iter(f"{{{SVG_NS}}}style")):
        style.getparent().remove(style)

    return etree.tostring(root, xml_declaration=True,
                           encoding="UTF-8").decode("utf-8")


def _parse_arc_path(d: str) -> dict | None:
    """Parse 'M x0 y0 A r r 0 0 1 x1 y1' → {cx, cy, r, t0, t1}.
    Assumes the form our pipeline emits (short arc, sweep=1)."""
    import math
    parts = _parse_d(d)
    if len(parts) < 2 or parts[0][0] != 'M' or parts[1][0] != 'A':
        return None
    x0, y0 = parts[0][1]
    rx, ry, _xrot, _large, sweep, x1, y1 = parts[1][1]
    # Reconstruct the arc center: it's equidistant from (x0,y0) and (x1,y1)
    # at distance rx. There are two candidate centers; pick by sweep.
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    dx, dy = x1 - x0, y1 - y0
    chord = math.hypot(dx, dy)
    if chord == 0 or rx == 0:
        return None
    h2 = rx * rx - (chord / 2) ** 2
    if h2 < 0:
        h2 = 0
    h = math.sqrt(h2)
    # Perpendicular unit vector
    nx, ny = -dy / chord, dx / chord
    if sweep == 1:
        cx, cy = mx - h * nx, my - h * ny
    else:
        cx, cy = mx + h * nx, my + h * ny
    t0 = math.atan2(y0 - cy, x0 - cx)
    t1 = math.atan2(y1 - cy, x1 - cx)
    return {"cx": cx, "cy": cy, "r": rx, "t0": t0, "t1": t1}


def _outline_per_letter(text_el, glyph_outline, upem, cap_h):
    """Convert a single-character <text> (with optional rotate transform)
    into a <g><path/></g> of the glyph outline."""
    from lxml import etree
    s = text_el.text or ""
    if not s:
        return None
    try:
        x = float(text_el.get("x", "0"))
        y = float(text_el.get("y", "0"))
        size = float(text_el.get("font-size", "16"))
    except ValueError:
        return None
    fill = text_el.get("fill", "black")
    text_anchor = text_el.get("text-anchor", "start")
    dom_baseline = text_el.get("dominant-baseline", "alphabetic")
    transform = text_el.get("transform", "")
    scale = size / upem

    # Compute total advance for text-anchor handling
    advances = []
    glyphs = []
    for ch in s:
        out = glyph_outline(ch)
        if out is None:
            glyphs.append(None)
            advances.append(0.0)
        else:
            d, adv = out
            glyphs.append(d)
            advances.append(adv * scale)
    total_w = sum(advances)
    if text_anchor == "middle":
        x_start = x - total_w / 2
    elif text_anchor == "end":
        x_start = x - total_w
    else:
        x_start = x
    # For dominant-baseline="central": visual center at y. Baseline below
    # by cap_h/2 (in font units → px).
    if dom_baseline in ("central", "middle"):
        baseline_y = y + (cap_h * scale) / 2
    else:
        baseline_y = y

    g = etree.Element(f"{{{SVG_NS}}}g", attrib={"fill": fill})
    if transform:
        g.set("transform", transform)
    cur_x = x_start
    for d, adv_px in zip(glyphs, advances):
        if d:
            t = f"translate({cur_x:.3f},{baseline_y:.3f}) scale({scale:.6f},{-scale:.6f})"
            etree.SubElement(g, f"{{{SVG_NS}}}path", d=d, transform=t)
        cur_x += adv_px
    return g


def _outline_textpath(text_el, textpath, arcs, glyph_outline, upem, cap_h, ttf_path):
    """Convert a <text><textPath href=#arc-N>...</textPath></text> into a
    <g> of glyph paths laid along the arc. Uses font advances to space
    characters, centered at the path's natural midpoint."""
    import math
    from lxml import etree
    from PIL import ImageFont

    href = textpath.get(f"{{{'http://www.w3.org/1999/xlink'}}}href") or textpath.get("href") or ""
    href = href.lstrip("#")
    arc = arcs.get(href)
    if not arc:
        return None

    s = textpath.text or ""
    if not s:
        return None
    size = float(text_el.get("font-size", "16"))
    fill = text_el.get("fill", "black")
    cx, cy, r, t0, t1 = arc["cx"], arc["cy"], arc["r"], arc["t0"], arc["t1"]

    # Determine direction: is_top = arc center below the text (sweep=1 in our
    # pipeline) → letters go in increasing theta. The pipeline always emits
    # arcs with sweep=1; the topmost point of the arc is at angle -π/2.
    is_top = True  # our pipeline only emits top-of-arc text via textPath
    direction = 1 if is_top else -1

    # Use PIL to get kerned advances
    font = ImageFont.truetype(ttf_path, int(round(size)))

    advances: list[float] = []
    prev = 0.0
    for i in range(len(s)):
        cur = float(font.getlength(s[: i + 1]))
        advances.append(cur - prev)
        prev = cur
    total_w = sum(advances)

    # Centered on the arc apex (-π/2 for top, π/2 for bottom).
    t_center = -math.pi / 2 if is_top else math.pi / 2

    scale = size / upem
    baseline_y_offset = (cap_h * scale) / 2  # dominant-baseline=central

    # x along the unwrapped text line (left edge to right edge)
    x_unwrap = 0.0
    g = etree.Element(f"{{{SVG_NS}}}g", attrib={"fill": fill})
    for ch, adv_px in zip(s, advances):
        center_unwrap = x_unwrap + adv_px / 2
        # Map center_unwrap to angular position
        theta = t_center + direction * (center_unwrap - total_w / 2) / r
        # Visual position of glyph center
        gx = cx + r * math.cos(theta)
        gy = cy + r * math.sin(theta)
        rot_deg = math.degrees(theta) + (90.0 if is_top else -90.0)

        # Compute glyph outline
        out = glyph_outline(ch)
        if out is None:
            x_unwrap += adv_px
            continue
        d, adv_units = out
        # Place glyph centered on (gx, gy) with rotation around that point,
        # accounting for dominant-baseline=central.
        # The glyph's natural left edge is at x=0 in font units; its center
        # along x is at adv_units/2. So translate by gx - adv_px/2 (in
        # post-scale coords), but the rotation is around (gx, gy), so it's
        # cleaner to first rotate then translate.
        tx = gx - (adv_px / 2)
        ty = gy + baseline_y_offset
        # Compose: rotate(rot_deg gx gy) translate(tx, ty) scale(scale, -scale)
        t = (f"rotate({rot_deg:.3f} {gx:.3f} {gy:.3f}) "
             f"translate({tx:.3f},{ty:.3f}) "
             f"scale({scale:.6f},{-scale:.6f})")
        etree.SubElement(g, f"{{{SVG_NS}}}path", d=d, transform=t)
        x_unwrap += adv_px
    return g


def svg_to_pdf_outlined(svg_path: Path | str, pdf_path: Path | str,
                         ttf_path: Path | str) -> None:
    """Outlined-text PDF — no font dependency in the output."""
    import cairosvg
    svg_text = Path(svg_path).read_text()
    converted = outline_text_in_svg(svg_text, str(ttf_path))
    cairosvg.svg2pdf(bytestring=converted.encode("utf-8"),
                      write_to=str(pdf_path))


# ---------- 2) EMBEDDED ----------

def svg_to_pdf_embedded(svg_path: Path | str, pdf_path: Path | str,
                         ttf_path: Path | str,
                         font_family: str = "Nourd-Bold") -> None:
    """Reportlab-based PDF where Nourd is embedded as a real TrueType font
    and text remains editable. Walks the SVG and emits equivalent reportlab
    draw calls."""
    from lxml import etree
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import math

    pdfmetrics.registerFont(TTFont(font_family, str(ttf_path)))
    root = etree.parse(str(svg_path)).getroot()
    W = float(root.get("width", "1000"))
    H = float(root.get("height", "1000"))

    c = canvas.Canvas(str(pdf_path), pagesize=(W, H))
    # Flip to SVG y-down coords once.
    c.translate(0, H)
    c.scale(1, -1)

    # Pre-index arc defs for textPath resolution
    arcs: dict[str, dict] = {}
    for p in root.iter(f"{{{SVG_NS}}}path"):
        pid = p.get("id", "")
        if pid.startswith("arc"):
            info = _parse_arc_path(p.get("d", ""))
            if info:
                arcs[pid] = info

    _walk_emit(c, root, arcs, font_family, ttf_path)
    c.save()


def _walk_emit(c, root, arcs, font_family, ttf_path):
    import math
    from lxml import etree
    from PIL import ImageFont

    # Stack of group transforms (id selectors etc.) — we don't need them
    # for our pipeline's flat structure, but support translate() on <g>
    # in case it ever appears.
    def emit(el):
        tag = etree.QName(el).localname
        if tag in ("defs", "style"):
            return  # skip
        if tag == "rect":
            x = float(el.get("x", 0))
            y = float(el.get("y", 0))
            w = float(el.get("width", 0))
            h = float(el.get("height", 0))
            fill = el.get("fill")
            if fill and fill != "none":
                c.saveState()
                c.setFillColor(fill)
                c.rect(x, y, w, h, stroke=0, fill=1)
                c.restoreState()
        elif tag == "circle":
            cx = float(el.get("cx"))
            cy = float(el.get("cy"))
            r = float(el.get("r"))
            stroke = el.get("stroke")
            fill = el.get("fill", "none")
            sw = float(el.get("stroke-width", 0))
            c.saveState()
            do_stroke = stroke and stroke != "none"
            do_fill = fill and fill != "none"
            if do_stroke:
                c.setStrokeColor(stroke)
                c.setLineWidth(sw)
            if do_fill:
                c.setFillColor(fill)
            if do_stroke or do_fill:
                c.circle(cx, cy, r,
                          stroke=1 if do_stroke else 0,
                          fill=1 if do_fill else 0)
            c.restoreState()
        elif tag == "line":
            x1 = float(el.get("x1"))
            y1 = float(el.get("y1"))
            x2 = float(el.get("x2"))
            y2 = float(el.get("y2"))
            stroke = el.get("stroke", "black")
            sw = float(el.get("stroke-width", 1))
            linecap = el.get("stroke-linecap", "butt")
            c.saveState()
            c.setStrokeColor(stroke)
            c.setLineWidth(sw)
            cap_map = {"butt": 0, "round": 1, "square": 2}
            c.setLineCap(cap_map.get(linecap, 0))
            c.line(x1, y1, x2, y2)
            c.restoreState()
        elif tag == "path":
            d = el.get("d", "")
            fill = el.get("fill", "black")
            stroke = el.get("stroke", "none")
            fill_rule = el.get("fill-rule", "nonzero")
            transform = el.get("transform", "")
            sw = float(el.get("stroke-width", 0))
            c.saveState()
            tx, ty, rot, rcx, rcy = _parse_transform(transform)
            if rot:
                c.translate(rcx, rcy); c.rotate(rot); c.translate(-rcx, -rcy)
            if tx or ty:
                c.translate(tx, ty)
            do_fill = fill and fill != "none"
            do_stroke = stroke and stroke != "none"
            if do_stroke:
                c.setStrokeColor(stroke); c.setLineWidth(sw)
            if do_fill:
                c.setFillColor(fill)
            p = c.beginPath()
            for cmd, args in _parse_d(d):
                u = cmd.upper()
                if u == "M": p.moveTo(args[0], args[1])
                elif u == "L": p.lineTo(args[0], args[1])
                elif u == "H": p.lineTo(args[0], p._currentPoint[1])
                elif u == "V": p.lineTo(p._currentPoint[0], args[0])
                elif u == "C": p.curveTo(args[0], args[1], args[2], args[3], args[4], args[5])
                elif u == "Q":
                    # Convert quadratic to cubic
                    x0, y0 = p._currentPoint
                    x1, y1, x2, y2 = args
                    cx1 = x0 + 2/3 * (x1 - x0); cy1 = y0 + 2/3 * (y1 - y0)
                    cx2 = x2 + 2/3 * (x1 - x2); cy2 = y2 + 2/3 * (y1 - y2)
                    p.curveTo(cx1, cy1, cx2, cy2, x2, y2)
                elif u == "Z": p.close()
                # A (arc) and S/T (smooth curves) not used by our pipeline
            if do_fill and do_stroke:
                c.drawPath(p, stroke=1, fill=1,
                            fillMode=(1 if fill_rule == "evenodd" else 0))
            elif do_fill:
                c.drawPath(p, stroke=0, fill=1,
                            fillMode=(1 if fill_rule == "evenodd" else 0))
            elif do_stroke:
                c.drawPath(p, stroke=1, fill=0)
            c.restoreState()
        elif tag == "text":
            _emit_text(c, el, arcs, font_family, ttf_path)
        elif tag == "g":
            transform = el.get("transform", "")
            tx, ty, rot, rcx, rcy = _parse_transform(transform)
            c.saveState()
            if rot:
                c.translate(rcx, rcy); c.rotate(rot); c.translate(-rcx, -rcy)
            if tx or ty:
                c.translate(tx, ty)
            for child in el:
                emit(child)
            c.restoreState()
        # Other tags are ignored

    for child in root:
        emit(child)


def _emit_text(c, text_el, arcs, font_family, ttf_path):
    """Emit a <text> element to the reportlab canvas, using the embedded
    Nourd font. Handles both per-letter (single char + transform=rotate)
    and textPath (text along an arc)."""
    import math
    from lxml import etree
    from reportlab.pdfbase import pdfmetrics
    from PIL import ImageFont

    textpath = text_el.find(f"{{{SVG_NS}}}textPath")
    fill = text_el.get("fill", "black")
    size = float(text_el.get("font-size", "16"))
    transform = text_el.get("transform", "")
    text_anchor = text_el.get("text-anchor", "start")
    dom_baseline = text_el.get("dominant-baseline", "alphabetic")

    # Reportlab cap-height retrieval for dominant-baseline=central
    face = pdfmetrics.getFont(font_family).face
    upem = face.unitsPerEm if hasattr(face, "unitsPerEm") else 1000
    cap_h = (getattr(face, "capHeight", None) or upem * 0.72)
    cap_h_px = (cap_h / upem) * size

    if textpath is None:
        # Per-letter or one-shot text element with x,y and optional rotate.
        s = text_el.text or ""
        if not s:
            return
        x = float(text_el.get("x", 0))
        y = float(text_el.get("y", 0))
        tx, ty, rot, rcx, rcy = _parse_transform(transform)
        c.saveState()
        if rot:
            c.translate(rcx, rcy); c.rotate(rot); c.translate(-rcx, -rcy)
        if tx or ty:
            c.translate(tx, ty)
        # Flip back so text renders right-side-up
        c.translate(x, y)
        c.scale(1, -1)
        c.setFillColor(fill)
        c.setFont(font_family, size)
        baseline_offset = -cap_h_px / 2 if dom_baseline in ("central", "middle") else 0
        if text_anchor == "middle":
            c.drawCentredString(0, baseline_offset, s)
        elif text_anchor == "end":
            c.drawRightString(0, baseline_offset, s)
        else:
            c.drawString(0, baseline_offset, s)
        c.restoreState()
        return

    # textPath: lay out characters along the arc
    s = textpath.text or ""
    if not s:
        return
    href = textpath.get(f"{{{'http://www.w3.org/1999/xlink'}}}href") or textpath.get("href") or ""
    href = href.lstrip("#")
    arc = arcs.get(href)
    if not arc:
        return
    cx, cy, r, t0, t1 = arc["cx"], arc["cy"], arc["r"], arc["t0"], arc["t1"]
    is_top = True  # our pipeline emits top-of-arc text only via textPath
    direction = 1 if is_top else -1
    font = ImageFont.truetype(str(ttf_path), int(round(size)))
    advances = []
    prev = 0.0
    for i in range(len(s)):
        cur = float(font.getlength(s[: i + 1]))
        advances.append(cur - prev)
        prev = cur
    total_w = sum(advances)
    t_center = -math.pi / 2 if is_top else math.pi / 2

    x_unwrap = 0.0
    for ch, adv_px in zip(s, advances):
        center_unwrap = x_unwrap + adv_px / 2
        theta = t_center + direction * (center_unwrap - total_w / 2) / r
        gx = cx + r * math.cos(theta)
        gy = cy + r * math.sin(theta)
        rot_deg = math.degrees(theta) + (90.0 if is_top else -90.0)
        c.saveState()
        c.translate(gx, gy)
        c.rotate(rot_deg)
        c.scale(1, -1)
        c.setFillColor(fill)
        c.setFont(font_family, size)
        baseline_offset = -cap_h_px / 2 if dom_baseline in ("central", "middle") else 0
        c.drawCentredString(0, baseline_offset, ch)
        c.restoreState()
        x_unwrap += adv_px


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: svg_to_pdf.py <mode:outlined|embedded> <input.svg> <output.pdf> [ttf]")
        sys.exit(1)
    mode = sys.argv[1]
    svg_in = sys.argv[2]
    pdf_out = sys.argv[3]
    ttf = sys.argv[4] if len(sys.argv) > 4 else "fonts/Nourd/nourd-700.ttf"
    if mode == "outlined":
        svg_to_pdf_outlined(svg_in, pdf_out, ttf)
    elif mode == "embedded":
        svg_to_pdf_embedded(svg_in, pdf_out, ttf)
    else:
        print(f"unknown mode: {mode}"); sys.exit(2)
    print(f"wrote {pdf_out} ({os.path.getsize(pdf_out)} bytes)")
