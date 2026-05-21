"""Assemble the final SVG: traced geometry as <path>, detected text as <text>."""
from __future__ import annotations

import math
from pathlib import Path
from xml.sax.saxutils import escape

from lxml import etree

from .types import DetectedText, FontMatch, TraceResult


SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"


def _arc_path_d(cx: float, cy: float, r: float, t0: float, t1: float) -> str:
    """Build an SVG arc path from (cx,cy,r) and angle endpoints in radians."""
    # Make sure we go CCW from t0 to t1 along the upper arc.
    if t1 < t0:
        t1 += 2 * math.pi
    x0 = cx + r * math.cos(t0)
    y0 = cy + r * math.sin(t0)
    x1 = cx + r * math.cos(t1)
    y1 = cy + r * math.sin(t1)
    large_arc = 1 if abs(t1 - t0) > math.pi else 0
    sweep = 1
    return f"M {x0:.2f} {y0:.2f} A {r:.2f} {r:.2f} 0 {large_arc} {sweep} {x1:.2f} {y1:.2f}"


def assemble(trace: TraceResult,
             texts: list[DetectedText],
             fonts: list[FontMatch],
             *,
             output_path: Path | str) -> Path:
    nsmap = {None: SVG_NS, "xlink": XLINK_NS}
    svg = etree.Element("svg", nsmap=nsmap)
    svg.set("viewBox", f"0 0 {trace.width} {trace.height}")
    svg.set("width", str(trace.width))
    svg.set("height", str(trace.height))

    defs = etree.SubElement(svg, "defs")

    # Background white rect (so the file rasterizes against white explicitly)
    bg = etree.SubElement(svg, "rect")
    bg.set("width", str(trace.width))
    bg.set("height", str(trace.height))
    bg.set("fill", "white")

    # Geometry
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

    # Text
    text_g = etree.SubElement(svg, "g")
    text_g.set("id", "text")
    for i, (dt, fm) in enumerate(zip(texts, fonts)):
        family = fm.family if fm else "sans-serif"
        weight = str(fm.weight) if fm else "700"
        size = f"{fm.size_px:.2f}" if fm and fm.size_px else "48"
        if dt.baseline and dt.baseline.kind == "arc":
            cx, cy, r, t0, t1 = dt.baseline.params
            arc_id = f"arc-{i}"
            arc = etree.SubElement(defs, "path")
            arc.set("id", arc_id)
            arc.set("d", _arc_path_d(cx, cy, r, t0, t1))
            arc.set("fill", "none")
            t = etree.SubElement(text_g, "text")
            t.set("text-anchor", "middle")
            t.set("fill", "black")
            t.set("font-family", family)
            t.set("font-weight", weight)
            t.set("font-size", size)
            tp = etree.SubElement(t, "textPath")
            tp.set("{%s}href" % XLINK_NS, f"#{arc_id}")
            tp.set("href", f"#{arc_id}")
            tp.set("startOffset", "50%")
            tp.text = dt.text
        else:
            # Straight text: place at polygon centroid, baseline-adjusted.
            x = float(dt.polygon[:, 0].mean())
            y = float(dt.polygon[:, 1].max()) - 4  # rough baseline
            t = etree.SubElement(text_g, "text")
            t.set("x", f"{x:.2f}")
            t.set("y", f"{y:.2f}")
            t.set("text-anchor", "middle")
            t.set("fill", "black")
            t.set("font-family", family)
            t.set("font-weight", weight)
            t.set("font-size", size)
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
