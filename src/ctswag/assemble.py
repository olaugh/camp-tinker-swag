"""Assemble the final SVG: traced geometry as <path>, detected text as <text>."""
from __future__ import annotations

import base64
import math
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
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
             circle_widths: list[float] | None = None,
             img_bgr: np.ndarray | None = None) -> Path:
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
    def _copy_element(src, dst_parent):
        """Append a deep-copied, namespace-stripped clone of src under dst_parent."""
        local = etree.QName(src).localname
        new = etree.SubElement(dst_parent, local)
        for k, v in src.attrib.items():
            new.set(etree.QName(k).localname, v)
        if src.text:
            new.text = src.text
        for child in src:
            _copy_element(child, new)

    for path_tag in trace.svg_paths:
        # Each entry is a serialized SVG snippet — usually a single
        # "<path .../>", but snap_strokes emits "<g><line/><circle/></g>"
        # when it needs asymmetric caps. Recursively reparent so the
        # children of any wrapper element come along too.
        try:
            el = etree.fromstring(path_tag)
            _copy_element(el, geom)
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
            # Subpixel sweep on the input asset showed `w - 0.5` (i.e.,
            # -1.5 px relative to the prior `w + 1.0` constant) maximizes
            # inside-ring SSIM. The previous overhang was tuned for
            # cairosvg's AA edge under-coverage; resvg handles AA
            # differently and prefers a slightly thinner stroke.
            c.set("stroke-width", f"{w - 0.5:.2f}")

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
        # ALGORITHMIC per-letter placement: when detection gave us arc
        # params (cx, cy, r) and cap heights, compute each character's
        # position from the chosen TEXT + FONT using natural advances.
        # Works for any text length — the same arc/font params can be
        # reused to regenerate the badge with different text.
        if (dt.letter_anchors and dt.baseline and dt.baseline.kind == "arc"
                and fm and fm.ttf_path):
            a0 = dt.letter_anchors[0]
            cx, cy, r, t0_b, t1_b = dt.baseline.params
            t_center = (t0_b + t1_b) / 2.0
            is_top = (a0["y"] < cy)
            from . import geom as _geom_mod
            text = dt.text or ""
            caps = [a["cap_h"] for a in dt.letter_anchors]
            # 0.8255 = 0.85 × 0.971; the 0.971 came from a subpixel SSIM
            # sweep on the final pipeline output, picking the size that
            # gives the best match against the original.
            common_size = (float(sorted(caps)[len(caps) // 2]) / 0.72) * 0.8255
            # Letter-spacing: t0_b/t1_b are the angles of the FIRST and
            # LAST characters' CENTERS (from per-letter detection). The
            # total text width must span CENTER-TO-CENTER of those, plus
            # half the first char's advance on the left and half the last
            # char's advance on the right (so the text envelope wraps
            # around them).
            #
            #   |--w_first/2--|-center-to-center span-|--w_last/2--|
            #   ^first char anchor             ^last char anchor
            #
            # center-to-center arc length = |t1 - t0| * r
            # target total_w = center-to-center + (w_first + w_last) / 2
            # letter_spacing = (target - natural_w) / n_pairs
            from PIL import ImageFont as _ImageFont
            _f = _ImageFont.truetype(fm.ttf_path, int(round(common_size)))
            natural_w = float(_f.getlength(text))
            w_first = float(_f.getlength(text[:1])) if text else 0.0
            w_last = float(_f.getlength(text[-1:])) if text else 0.0
            center_to_center_arc = abs(t1_b - t0_b) * r
            target_total_w = center_to_center_arc + (w_first + w_last) / 2
            n_pairs = max(len(text) - 1, 1)
            letter_spacing_px = max(0.0, (target_total_w - natural_w) / n_pairs)
            # Shift t_center to account for asymmetric first/last char
            # widths. Derived from the two equations
            #   t_first_anchor = t_center - total_w/(2r) + w_first/(2r)
            #   t_last_anchor  = t_center + total_w/(2r) - w_last/(2r)
            # summed gives t_center = midpoint + (w_last - w_first)/(4r)
            # (NOT /2r — that was wrong by a factor of 2 and shifted the
            # text twice as far as needed when first/last had different
            # advances, e.g. "2" vs "5" in 2025).
            t_center_shift = (w_last - w_first) / (4 * r)
            t_center_corr = t_center + (t_center_shift if is_top else -t_center_shift)
            placements = _geom_mod.place_text_on_arc(
                text, fm.ttf_path, common_size,
                cx, cy + dy, r, t_center_corr, is_top=is_top,
                letter_spacing_px=letter_spacing_px,
            )
            if placements:
                # PER-CHARACTER REFINEMENT: small dx/dy sweep around each
                # algorithm-computed position, maximizing IoU within the
                # original letter's mask. Size and rotation stay locked
                # (algorithm provides those). Sweep is bounded to ±10 px
                # so a stable algorithmic placement isn't undone by mask
                # noise. Skipped if img_bgr isn't passed (e.g., regen
                # from saved arc params without a reference image).
                if img_bgr is not None:
                    anchors_by_char_idx = list(dt.letter_anchors)  # 1:1 with placements
                    for i, p in enumerate(placements):
                        if i >= len(anchors_by_char_idx):
                            break
                        mask = anchors_by_char_idx[i].get("mask")
                        if mask is None:
                            continue
                        best_x, best_y, best_rot, _ = _geom_mod.sweep_char_anchor(
                            img_bgr, fm.ttf_path, common_size, p["char"],
                            p["x"], p["y"], p["rot_deg"], mask,
                            search_radius=10.0,
                            rot_search_deg=2.0,
                        )
                        p["x"], p["y"] = best_x, best_y
                        p["rot_deg"] = best_rot
                for p in placements:
                    ax, ay = p["x"] + dx, p["y"]
                    t = etree.SubElement(text_g, "text")
                    t.set("x", f"{ax:.2f}")
                    t.set("y", f"{ay:.2f}")
                    t.set("text-anchor", "middle")
                    t.set("dominant-baseline", "central")
                    t.set("fill", "black")
                    t.set("font-family", family)
                    t.set("font-weight", weight)
                    t.set("font-size", f"{common_size:.2f}")
                    t.set("transform",
                          f"rotate({p['rot_deg']:.2f} {ax:.2f} {ay:.2f})")
                    t.text = p["char"]
                continue  # skip the textPath fallback below
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
