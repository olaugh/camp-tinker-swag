"""Assemble the final SVG: traced geometry as <path>, detected text as <text>."""
from __future__ import annotations

import base64
import math
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

from .types import DetectedText, FontMatch, TraceResult


def _ttf_family_name(ttf_path: str) -> str | None:
    """Return the font's family name as written in the TTF's `name` table.

    Without this, the SVG carries our corpus-directory-derived name (e.g.
    "BrandonGrotesque") but fontconfig knows the font by its TTF-table name
    (e.g. "BrandonGrotesque-Bold"). rsvg-convert and other fontconfig-based
    renderers fail to match the name we emit and silently fall back. Using
    the TTF's real family name everywhere keeps the SVG portable.
    """
    try:
        from fontTools.ttLib import TTFont
    except Exception:
        return None
    try:
        f = TTFont(ttf_path)
        n = f["name"]
        # Try the Windows English entry first, then Mac Roman.
        rec = n.getName(1, 3, 1, 0x409) or n.getName(1, 1, 0, 0)
        if rec is None:
            return None
        return rec.toUnicode()
    except Exception:
        return None


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _font_face_css(fonts: list[FontMatch]) -> str:
    """Build @font-face CSS rules embedding each used TTF as a base64
    data-URL. The font-family in each rule matches what `font-family`
    on the <text> elements uses (the TTF's real family name, see
    _ttf_family_name), so:
      * Browsers like Chrome use the embedded TTF via @font-face match.
      * rsvg-convert (which ignores @font-face data URLs) instead loads
        the same TTF by name through fontconfig -- we register
        fonts_curated/ in a project-local fontconfig file at run time.
    """
    seen: set[tuple[str, str]] = set()
    rules: list[str] = []
    for fm in fonts:
        if not fm or not fm.ttf_path:
            continue
        family = _ttf_family_name(fm.ttf_path) or fm.family
        key = (family, fm.ttf_path)
        if key in seen:
            continue
        seen.add(key)
        try:
            with open(fm.ttf_path, "rb") as f:
                ttf_bytes = f.read()
        except OSError:
            continue
        b64 = base64.b64encode(ttf_bytes).decode("ascii")
        rules.append(
            f"@font-face {{\n"
            f"  font-family: '{family}';\n"
            f"  font-weight: {fm.weight};\n"
            f"  font-style: normal;\n"
            f"  src: url(data:font/ttf;base64,{b64}) format('truetype');\n"
            f"}}"
        )
    return "\n".join(rules)


def _arc_path_d(cx: float, cy: float, r: float, t0: float, t1: float) -> str:
    """Build an SVG arc path from (cx,cy,r) and angle endpoints in radians.

    Always picks the SHORT arc between the two endpoints (large_arc=0). The
    sweep flag is chosen so the path is traversed in the direction of
    increasing parameter t (top arcs run left->right with increasing theta;
    bottom arcs run left->right with DECREASING theta, so they need
    sweep=0). Calling code orders (t0, t1) so the FIRST point is the text's
    leftmost end -- the textPath then lays glyphs in reading order.
    """
    # Normalize delta to (-pi, pi] so we know whether t0->t1 sweeps in the
    # positive (sweep=1, CW in image coords) or negative direction.
    delta = t1 - t0
    while delta > math.pi:
        delta -= 2 * math.pi
    while delta <= -math.pi:
        delta += 2 * math.pi
    sweep = 1 if delta > 0 else 0
    large_arc = 0
    x0 = cx + r * math.cos(t0)
    y0 = cy + r * math.sin(t0)
    x1 = cx + r * math.cos(t1)
    y1 = cy + r * math.sin(t1)
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {large_arc} {sweep} {x1:.2f} {y1:.2f}"


def assemble(trace: TraceResult,
             texts: list[DetectedText],
             fonts: list[FontMatch],
             *,
             output_path: Path | str,
             circles: list[tuple[float, float, float]] | None = None,
             circle_widths: list[float] | None = None) -> Path:
    nsmap = {None: SVG_NS, "xlink": XLINK_NS}
    svg = etree.Element("svg", nsmap=nsmap)
    svg.set("viewBox", f"0 0 {trace.width} {trace.height}")
    svg.set("width", str(trace.width))
    svg.set("height", str(trace.height))

    defs = etree.SubElement(svg, "defs")

    # Embed each used font as a base64 @font-face so any renderer draws the
    # actual font without needing it installed. Without this, browsers fall
    # back to a default sans (or serif!) while cairosvg, which finds the
    # TTF via fontconfig in the project, renders correctly -- producing a
    # mismatch between the live SVG and the render.png.
    font_css = _font_face_css(fonts)
    if font_css:
        style = etree.SubElement(defs, "style")
        style.set("type", "text/css")
        style.text = font_css

    # Background white rect (so the file rasterizes against white explicitly)
    bg = etree.SubElement(svg, "rect")
    bg.set("width", str(trace.width))
    bg.set("height", str(trace.height))
    bg.set("fill", "white")

    # Geometry FIRST so the rings render on top of it. In the source, the
    # tips of line-art strokes (sun rays, mountain ridges, the baseline)
    # actually continue a few pixels under the ring's stroke -- visually
    # they "end at the ring" but the ink continues hidden beneath. Painting
    # the ring on top reproduces that, and also hides any sub-pixel
    # anti-alias halo from the ring's edges that vtracer might trace as
    # faint fragments.
    geom = etree.SubElement(svg, "g")
    geom.set("id", "geometry")
    geom.set("fill", "black")
    geom.set("stroke", "none")
    for path_tag in trace.svg_paths:
        # Each entry is a serialized "<path d='...' transform='...' fill='...'/>" string.
        # Parse and reattach without namespace.
        try:
            el = etree.fromstring(path_tag)
            # strip namespace
            local = etree.QName(el).localname
            new = etree.SubElement(geom, local)
            for k, v in el.attrib.items():
                new.set(etree.QName(k).localname, v)
        except etree.XMLSyntaxError:
            # If parsing fails, treat as a raw `d` string
            p = etree.SubElement(geom, "path")
            p.set("d", path_tag)

    # Rings on top of geometry. See note above the geometry block.
    # The drawn stroke-width is the detected stroke + 1 px: cairosvg's AA
    # under-covers pixels at the very inner/outer edge of a stroked circle
    # by ~0.5 px on each side, which would leave a 1-px halo between the
    # line art (vtracer-traced) and the SVG ring. The +1 px overhang
    # closes that halo without visibly changing the ring's thickness.
    if circles:
        ring_g = etree.SubElement(svg, "g")
        ring_g.set("id", "rings")
        ring_g.set("fill", "none")
        ring_g.set("stroke", "black")
        widths = circle_widths or [6.0] * len(circles)
        for (cx, cy, r), w in zip(circles, widths):
            c = etree.SubElement(ring_g, "circle")
            c.set("cx", f"{cx:.2f}")
            c.set("cy", f"{cy:.2f}")
            c.set("r", f"{r:.2f}")
            c.set("stroke-width", f"{w + 1.0:.2f}")

    # Text
    text_g = etree.SubElement(svg, "g")
    text_g.set("id", "text")
    for i, (dt, fm) in enumerate(zip(texts, fonts)):
        # Use the TTF's REAL family name (from its `name` table) so
        # fontconfig-based renderers (rsvg-convert, Inkscape) can resolve
        # the font. The @font-face block above declares the same name with
        # the TTF embedded as a data URL for browsers.
        family = ((_ttf_family_name(fm.ttf_path) if fm and fm.ttf_path else None)
                   or (fm.family if fm else "sans-serif"))
        weight = str(fm.weight) if fm else "700"
        size = f"{fm.size_px:.2f}" if fm and fm.size_px else "48"
        ls_em = fm.letter_spacing_em if fm else 0.0
        dx = fm.dx if fm else 0.0
        dy = fm.dy if fm else 0.0
        if dt.baseline and dt.baseline.kind == "arc":
            cx, cy, r, t0, t1 = dt.baseline.params
            # Apply optimiser dy as a vertical shift of the arc center
            cy_eff = cy + dy
            arc_id = f"arc-{i}"
            arc = etree.SubElement(defs, "path")
            arc.set("id", arc_id)
            arc.set("d", _arc_path_d(cx, cy_eff, r, t0, t1))
            arc.set("fill", "none")
            t = etree.SubElement(text_g, "text")
            t.set("text-anchor", "middle")
            t.set("fill", "black")
            t.set("font-family", family)
            t.set("font-weight", weight)
            t.set("font-size", size)
            if ls_em:
                # SVG letter-spacing is in length units (px) when given without unit
                t.set("letter-spacing", f"{ls_em * float(size):.2f}")
            tp = etree.SubElement(t, "textPath")
            tp.set("{%s}href" % XLINK_NS, f"#{arc_id}")
            tp.set("href", f"#{arc_id}")
            # dx along the arc -> shift startOffset (linear along-arc length)
            arc_len = abs(t1 - t0) * r  # rough arc length
            if arc_len > 0:
                offset_pct = 50.0 + (dx / arc_len) * 100.0
                tp.set("startOffset", f"{offset_pct:.2f}%")
            else:
                tp.set("startOffset", "50%")
            tp.text = dt.text
        else:
            # Straight text: place at polygon centroid, baseline-adjusted.
            x = float(dt.polygon[:, 0].mean()) + dx
            y = float(dt.polygon[:, 1].max()) - 4 + dy
            t = etree.SubElement(text_g, "text")
            t.set("x", f"{x:.2f}")
            t.set("y", f"{y:.2f}")
            t.set("text-anchor", "middle")
            t.set("fill", "black")
            t.set("font-family", family)
            t.set("font-weight", weight)
            t.set("font-size", size)
            if ls_em:
                t.set("letter-spacing", f"{ls_em * float(size):.2f}")
            t.text = dt.text

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree = etree.ElementTree(svg)
    tree.write(str(output_path), pretty_print=True, xml_declaration=True, encoding="utf-8")
    return output_path


def to_print_mode(svg_path: Path | str, out_path: Path | str,
                  family_to_ttf: dict[str, str]) -> Path:
    """Convert <text>/<textPath> nodes to <path> via fontTools SVGPathPen.

    Output is the same SVG with text replaced by black filled paths.
    Suitable for screen printers.
    """
    from fontTools.ttLib import TTFont
    from fontTools.pens.svgPathPen import SVGPathPen
    svg_path = Path(svg_path)
    tree = etree.parse(str(svg_path))
    root = tree.getroot()

    # Index TTFonts by family
    fonts = {}
    for fam, p in family_to_ttf.items():
        fonts[fam] = TTFont(p)

    # Find every <text>; for each, render to paths.
    ns = root.nsmap.get(None, SVG_NS)
    text_xpath = f".//{{{ns}}}text"
    for text_el in list(root.findall(text_xpath)):
        family = text_el.get("font-family", "")
        size = float(text_el.get("font-size", "48"))
        # For now we just outline the literal text content (no textPath warping).
        string = (text_el.text or "").strip()
        if not string and len(text_el) > 0:
            # textPath case
            for child in text_el:
                tag = etree.QName(child).localname
                if tag == "textPath":
                    string = (child.text or "").strip()
                    break
        if not string or family not in fonts:
            continue
        # Place flat at the text element's (x,y) anchor.
        x = float(text_el.get("x", "0"))
        y = float(text_el.get("y", "0"))
        ttf = fonts[family]
        cmap = ttf.getBestCmap()
        glyph_set = ttf.getGlyphSet()
        units_per_em = ttf["head"].unitsPerEm
        scale = size / units_per_em
        pen = SVGPathPen(glyph_set)
        adv_x = 0.0
        for ch in string:
            gid = cmap.get(ord(ch))
            if gid is None:
                continue
            g = glyph_set[gid]
            pen.path = []  # not used; we'll re-pen per glyph
            sub_pen = SVGPathPen(glyph_set)
            g.draw(sub_pen)
            d = sub_pen.getCommands()
            # transform: scale, then translate. SVG y-flip needed (fontTools is y-up).
            transformed = (
                f'<g transform="translate({x + adv_x},{y}) scale({scale},{-scale})">'
                f'<path d="{d}" fill="black"/></g>'
            )
            new_el = etree.fromstring(transformed)
            text_el.addnext(new_el)
            adv_x += g.width * scale
        # remove the original text element
        text_el.getparent().remove(text_el)

    out_path = Path(out_path)
    tree.write(str(out_path), pretty_print=True, xml_declaration=True, encoding="utf-8")
    return out_path
