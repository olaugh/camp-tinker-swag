"""Synthesize a Camp-Tinker-style badge for development & ground-truth eval.

The user's actual reference image isn't checked in yet; this generator
produces a similar B/W badge that lets the pipeline be developed and
evaluated against a *known* font and geometry.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import cairosvg
import numpy as np
from PIL import Image


BADGE_SVG_TEMPLATE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
  <defs>
    <path id="arc-top" d="M {arc_x0} {arc_y0} A {arc_r} {arc_r} 0 0 1 {arc_x1} {arc_y1}" fill="none"/>
  </defs>

  <!-- background -->
  <rect width="{W}" height="{H}" fill="white"/>

  <!-- outer + inner rings -->
  <circle cx="{cx}" cy="{cy}" r="{r_outer}" fill="none" stroke="black" stroke-width="{ring_sw}"/>
  <circle cx="{cx}" cy="{cy}" r="{r_inner}" fill="none" stroke="black" stroke-width="{ring_sw}"/>

  <!-- sun rays (radial lines inside the upper half) -->
  <g stroke="black" stroke-width="{ray_sw}" stroke-linecap="butt">
    {rays}
  </g>

  <!-- semicircular sun arc -->
  <path d="M {sun_x0} {sun_y} A {sun_r} {sun_r} 0 0 1 {sun_x1} {sun_y}" fill="white" stroke="black" stroke-width="{ring_sw}"/>

  <!-- three mountains -->
  <polyline points="{mountains}" fill="white" stroke="black" stroke-width="{ring_sw}" stroke-linejoin="miter"/>

  <!-- snow cap on central peak -->
  <polyline points="{snow}" fill="none" stroke="black" stroke-width="{ring_sw}"/>

  <!-- baseline strip with vertical strokes (waterfall/wood) -->
  <line x1="{base_x0}" y1="{base_y}" x2="{base_x1}" y2="{base_y}" stroke="black" stroke-width="{ring_sw}"/>
  <g stroke="black" stroke-width="{tick_sw}" stroke-linecap="round">
    {ticks}
  </g>

  <!-- two pine trees -->
  {pines}

  <!-- arched text "CAMP TINKER" -->
  <text text-anchor="middle" fill="black"
        font-family="{font_family}" font-weight="{weight}"
        font-size="{title_px}" letter-spacing="{ls_px}">
    <textPath href="#arc-top" startOffset="50%">{title}</textPath>
  </text>

  <!-- straight year text -->
  <text x="{cx}" y="{year_y}" text-anchor="middle" fill="black"
        font-family="{font_family}" font-weight="{weight}"
        font-size="{year_px}" letter-spacing="{ls_px}">{year}</text>
</svg>"""


@dataclass
class SynthBadge:
    """Bundles the rendered PNG with everything we know about it."""
    png_bytes: bytes
    svg_str: str
    font_family: str
    weight: int
    title_px: float
    year_px: float
    arc_center: tuple[float, float]
    arc_radius: float
    year_pos: tuple[float, float]
    width: int
    height: int
    title_text: str
    year_text: str


def _pine(cx: float, cy: float, h: float, sw: float) -> str:
    """Return SVG group for a stylized pine tree."""
    half = h / 2.5
    trunk_w = sw * 0.5
    parts = []
    # triangular tiers
    for i, frac in enumerate([0.0, 0.25, 0.5]):
        top_y = cy - h * (1 - frac)
        base_y = cy - h * (0.5 - frac)
        width = half * (0.6 + 0.4 * frac)
        parts.append(
            f'<polygon points="{cx},{top_y} {cx-width},{base_y} {cx+width},{base_y}" '
            f'fill="white" stroke="black" stroke-width="{sw}"/>'
        )
    # trunk
    parts.append(
        f'<rect x="{cx-trunk_w}" y="{cy-h*0.05}" width="{trunk_w*2}" height="{h*0.1}" '
        f'fill="black"/>'
    )
    return "<g>" + "\n".join(parts) + "</g>"


def synthesize(
    *,
    font_family: str = "Montserrat",
    weight: int = 800,
    title_text: str = "CAMP TINKER",
    year_text: str = "2025",
    size: int = 1024,
    seed: int = 0,
) -> SynthBadge:
    rng = np.random.default_rng(seed)
    W = H = size
    cx, cy = W / 2, H / 2 + size * 0.05
    r_outer = size * 0.34
    r_inner = r_outer - size * 0.015
    ring_sw = size * 0.006
    ray_sw = size * 0.0045
    tick_sw = size * 0.005

    # rays inside upper half
    rays_inner_r = r_inner - size * 0.02
    rays_outer_r = r_inner - size * 0.18  # the rays end well inside the ring
    n_rays = 36
    ray_strs = []
    for i in range(n_rays):
        # spread rays across upper half (180..360 deg in screen coords = above center)
        theta = math.pi + (i + 0.5) / n_rays * math.pi
        # only keep those above the sun arc (center top region)
        x1 = cx + rays_inner_r * math.cos(theta)
        y1 = cy + rays_inner_r * math.sin(theta)
        x2 = cx + rays_outer_r * math.cos(theta)
        y2 = cy + rays_outer_r * math.sin(theta)
        ray_strs.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')

    # sun arc behind mountains
    sun_r = size * 0.13
    sun_y = cy + size * 0.05
    sun_x0 = cx - sun_r
    sun_x1 = cx + sun_r

    # mountains
    base_y = cy + size * 0.12
    base_x0 = cx - r_inner + size * 0.03
    base_x1 = cx + r_inner - size * 0.03
    peak_y = cy - size * 0.08
    side_peak_y = cy + size * 0.02
    mountains = " ".join([
        f"{base_x0},{base_y}",
        f"{cx - size*0.18},{side_peak_y}",
        f"{cx - size*0.07},{base_y - size*0.02}",
        f"{cx},{peak_y}",
        f"{cx + size*0.07},{base_y - size*0.02}",
        f"{cx + size*0.18},{side_peak_y}",
        f"{base_x1},{base_y}",
    ])
    snow = " ".join([
        f"{cx - size*0.025},{peak_y + size*0.04}",
        f"{cx - size*0.015},{peak_y + size*0.03}",
        f"{cx - size*0.005},{peak_y + size*0.045}",
        f"{cx + size*0.005},{peak_y + size*0.03}",
        f"{cx + size*0.015},{peak_y + size*0.045}",
        f"{cx + size*0.025},{peak_y + size*0.03}",
    ])

    # waterfall ticks
    n_ticks = 12
    tick_strs = []
    for i in range(n_ticks):
        tx = cx + (i - n_ticks/2 + 0.5) * size * 0.02
        ty0 = base_y + size * 0.005
        ty1 = base_y + size * 0.05 + rng.uniform(-size * 0.005, size * 0.005)
        tick_strs.append(f'<line x1="{tx:.1f}" y1="{ty0:.1f}" x2="{tx:.1f}" y2="{ty1:.1f}"/>')

    # pines
    pine_h = size * 0.12
    pines = "\n".join([
        _pine(cx - r_inner + size * 0.06, base_y + size * 0.02, pine_h, ring_sw),
        _pine(cx + r_inner - size * 0.06, base_y + size * 0.02, pine_h, ring_sw),
    ])

    # arched text: above the outer ring
    arc_r = r_outer + size * 0.06
    arc_span = math.radians(140)  # 140 deg arc
    theta0 = -math.pi/2 - arc_span/2
    theta1 = -math.pi/2 + arc_span/2
    arc_x0 = cx + arc_r * math.cos(theta0)
    arc_y0 = cy + arc_r * math.sin(theta0)
    arc_x1 = cx + arc_r * math.cos(theta1)
    arc_y1 = cy + arc_r * math.sin(theta1)

    # year position: below the outer ring
    year_y = cy + r_outer + size * 0.08

    title_px = size * 0.075
    year_px = size * 0.07

    svg = BADGE_SVG_TEMPLATE.format(
        W=W, H=H, cx=cx, cy=cy,
        r_outer=r_outer, r_inner=r_inner, ring_sw=ring_sw, ray_sw=ray_sw,
        rays="\n    ".join(ray_strs),
        sun_x0=sun_x0, sun_x1=sun_x1, sun_y=sun_y, sun_r=sun_r,
        mountains=mountains, snow=snow,
        base_x0=base_x0, base_x1=base_x1, base_y=base_y,
        tick_sw=tick_sw, ticks="\n    ".join(tick_strs),
        pines=pines,
        arc_x0=arc_x0, arc_y0=arc_y0, arc_x1=arc_x1, arc_y1=arc_y1, arc_r=arc_r,
        font_family=font_family,
        weight=weight, title_px=title_px, year_px=year_px, ls_px=title_px * 0.05,
        title=title_text, year=year_text, year_y=year_y,
    )

    # Render via cairosvg
    png_bytes = cairosvg.svg2png(bytestring=svg.encode("utf-8"), output_width=W, output_height=H)
    return SynthBadge(
        png_bytes=png_bytes, svg_str=svg, font_family=font_family, weight=weight,
        title_px=title_px, year_px=year_px,
        arc_center=(cx, cy), arc_radius=arc_r,
        year_pos=(cx, year_y), width=W, height=H,
        title_text=title_text, year_text=year_text,
    )


def write_synth(out_dir: Path | str, name: str = "camp_tinker_2025_synth", **kw) -> SynthBadge:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    badge = synthesize(**kw)
    (out_dir / f"{name}.png").write_bytes(badge.png_bytes)
    (out_dir / f"{name}.svg").write_text(badge.svg_str)
    return badge


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="assets")
    p.add_argument("--name", default="camp_tinker_2025_synth")
    p.add_argument("--font", default="Montserrat")
    p.add_argument("--weight", type=int, default=800)
    p.add_argument("--size", type=int, default=1024)
    args = p.parse_args()
    b = write_synth(args.out, args.name, font_family=args.font, weight=args.weight, size=args.size)
    print(f"wrote {args.out}/{args.name}.png  ({b.width}x{b.height}, {b.font_family} {b.weight})")
