"""Fetch a wide font corpus from the Fontsource catalog.

By default grabs every sans-serif and display family that has Latin
subset support, picking up to 4 representative weights per family
(400, 600, 700, 800 -- skipping italic/condensed variants for the
first pass). Idempotent: any TTF already on disk is skipped.

Usage:
    .venv/bin/python scripts/fetch_fonts_full.py
    .venv/bin/python scripts/fetch_fonts_full.py --categories sans-serif --max-families 500
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


CATALOG_URL = "https://api.fontsource.org/v1/fonts"
CDN = "https://cdn.jsdelivr.net/fontsource/fonts"

DEFAULT_WEIGHTS = [400, 500, 600, 700, 800, 900]


def fetch_catalog(categories: list[str]) -> list[dict]:
    """Returns Fontsource catalog entries filtered to the requested categories."""
    cat_param = ",".join(categories)
    url = f"{CATALOG_URL}?subsets=latin&category={cat_param}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def cdn_url(slug: str, weight: int) -> str:
    return f"{CDN}/{slug}@latest/latin-{weight}-normal.ttf"


def download(url: str, dest: Path) -> bool:
    if dest.exists():
        return False
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        print(f"  fail {url}: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  fail {url}: {e}", file=sys.stderr)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=Path(__file__).resolve().parent.parent / "fonts",
                   type=Path)
    p.add_argument("--categories", default="sans-serif,display",
                   help="comma-separated Fontsource categories")
    p.add_argument("--max-families", type=int, default=10_000)
    p.add_argument("--max-weights", type=int, default=4,
                   help="Cap weights per family to keep the corpus manageable.")
    p.add_argument("--workers", type=int, default=16)
    args = p.parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]
    catalog = fetch_catalog(categories)
    print(f"catalog: {len(catalog)} families ({categories})")
    catalog = catalog[: args.max_families]

    # Pick weights present in family, biased toward bolds (which our test asset uses).
    targets: list[tuple[str, str, int]] = []  # (family_dir_name, slug, weight)
    for ent in catalog:
        slug = ent["id"]
        family_dir = ent["family"].replace(" ", "")
        weights = sorted(set(ent.get("weights", [])) & set(DEFAULT_WEIGHTS))
        # Prefer heavier weights (700/800/900) for badge-style text.
        weights.sort(key=lambda w: (-w if w >= 700 else 1000 - w))
        for w in weights[: args.max_weights]:
            targets.append((family_dir, slug, w))

    print(f"queued {len(targets)} (family, weight) downloads → {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    new = 0
    fails = 0
    seen = set()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for family_dir, slug, w in targets:
            dest = args.out / family_dir / f"{slug}-{w}.ttf"
            key = str(dest)
            if key in seen:
                continue
            seen.add(key)
            url = cdn_url(slug, w)
            futs[ex.submit(download, url, dest)] = (family_dir, slug, w)
        for i, f in enumerate(as_completed(futs)):
            family_dir, slug, w = futs[f]
            try:
                if f.result():
                    new += 1
                else:
                    fails += 1
            except Exception:
                fails += 1
            if (i + 1) % 200 == 0:
                print(f"  ... {i+1}/{len(futs)} done")

    total = sum(1 for _ in args.out.rglob("*.ttf"))
    print(f"done. +{new} new, {fails} skipped/failed, {total} total TTFs under {args.out}")


if __name__ == "__main__":
    main()
