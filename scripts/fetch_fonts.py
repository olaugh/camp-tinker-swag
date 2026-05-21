"""Fetch a small Google Fonts corpus for font-ID render-and-diff.

Uses the Fontsource CDN (jsDelivr) which ships one static TTF per
(family, weight). One file per weight makes the corpus iterable
without needing to instantiate variable fonts at runtime.

Idempotent: skips files that already exist.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

CDN = "https://cdn.jsdelivr.net/fontsource/fonts"
# (fontsource slug, [list of weights to download])
CORPUS: dict[str, tuple[str, list[int]]] = {
    "Montserrat":    ("montserrat",     [400, 500, 600, 700, 800, 900]),
    "Poppins":       ("poppins",        [400, 500, 600, 700, 800, 900]),
    "Raleway":       ("raleway",        [400, 600, 700, 800, 900]),
    "Nunito":        ("nunito",         [400, 600, 700, 800, 900]),
    "Quicksand":     ("quicksand",      [400, 500, 600, 700]),
    "WorkSans":      ("work-sans",      [400, 600, 700, 800, 900]),
    "Inter":         ("inter",          [400, 600, 700, 800, 900]),
    "Manrope":       ("manrope",        [400, 600, 700, 800]),
    "Lato":          ("lato",           [400, 700, 900]),
    "OpenSans":      ("open-sans",      [400, 600, 700, 800]),
    "Oswald":        ("oswald",         [400, 500, 600, 700]),
    "BebasNeue":     ("bebas-neue",     [400]),
    "LeagueSpartan": ("league-spartan", [400, 600, 700, 800, 900]),
    "Archivo":       ("archivo",        [400, 600, 700, 800, 900]),
    "Barlow":        ("barlow",         [400, 600, 700, 800, 900]),
}


def fetch_one(slug: str, weight: int, dest: Path) -> Path | None:
    out = dest / f"{slug}-{weight}.ttf"
    if out.exists():
        return None
    url = f"{CDN}/{slug}@latest/latin-{weight}-normal.ttf"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        print(f"  fail {slug} {weight}: {e}", file=sys.stderr)
        return None
    out.write_bytes(data)
    return out


def main():
    root = Path(__file__).resolve().parent.parent / "fonts"
    root.mkdir(exist_ok=True)
    total_new = 0
    for family, (slug, weights) in CORPUS.items():
        sub = root / family
        sub.mkdir(parents=True, exist_ok=True)
        new = 0
        for w in weights:
            if fetch_one(slug, w, sub):
                new += 1
        existing = sum(1 for _ in sub.glob("*.ttf"))
        total_new += new
        print(f"  {family:<16} +{new} ({existing} total)")
    print(f"done. {total_new} new TTFs under {root}")


if __name__ == "__main__":
    main()
