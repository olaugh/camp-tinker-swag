"""Polite scrape of Fontshare + League of Moveable Type, deduped against fonts/.

Both sources allow downloads. We:
- identify ourselves with a User-Agent that includes contact info
- sleep ≥ POLITE_DELAY seconds between requests to the same host
- single-threaded (no parallel hammering of one origin)
- skip families that already exist in fonts/ (normalized name match)
- extract per-weight TTFs and rename to <slug>-<weight>.ttf so fontid.discover_corpus
  picks them up via the existing weight-suffix regex.

Font Squirrel is intentionally NOT scraped: their CloudFront WAF returns a bot
challenge (x-amzn-waf-action: challenge) on plain GET requests. Respect that.

Usage:
    .venv/bin/python scripts/fetch_fonts_extra.py
    .venv/bin/python scripts/fetch_fonts_extra.py --source fontshare
    .venv/bin/python scripts/fetch_fonts_extra.py --source lotm
    .venv/bin/python scripts/fetch_fonts_extra.py --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path

USER_AGENT = "ctswag-corpus-bot/0.1 (research; olaughlin@gmail.com)"
POLITE_DELAY = 2.0  # seconds between requests to the same host

FONTSHARE_LIST = "https://api.fontshare.com/v2/fonts?offset=0&limit=200"
FONTSHARE_DOWNLOAD = "https://api.fontshare.com/v2/fonts/download/{slug}"

GITHUB_API = "https://api.github.com"
LOTM_ORG = "theleagueof"

# Weights we keep (matches the existing corpus convention).
KEEP_WEIGHTS = {100, 200, 300, 400, 500, 600, 700, 800, 900}

# Fontshare's weight numbers come straight from the API; LOTM zips name their
# files like "Orbitron-Bold.ttf" so we map by stem suffix.
LOTM_WEIGHT_NAMES = {
    "thin": 100,
    "extralight": 200,
    "ultralight": 200,
    "light": 300,
    "regular": 400,
    "normal": 400,
    "book": 400,
    "medium": 500,
    "demibold": 600,
    "semibold": 600,
    "bold": 700,
    "extrabold": 800,
    "ultrabold": 800,
    "heavy": 900,
    "black": 900,
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def http_get(url: str, *, accept: str | None = None) -> bytes:
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        enc = resp.headers.get("Content-Encoding", "").lower()
        if enc == "gzip":
            import gzip
            data = gzip.decompress(data)
        elif enc == "deflate":
            import zlib
            data = zlib.decompress(data)
        return data


def family_dir_name(name: str) -> str:
    # Match the existing corpus convention: PascalCase with no spaces.
    parts = re.split(r"[\s_-]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def write_ttf(out_dir: Path, slug: str, weight: int, data: bytes, *, dry_run: bool) -> bool:
    dest = out_dir / f"{slug}-{weight}.ttf"
    if dest.exists():
        return False
    if dry_run:
        print(f"    [dry-run] would write {dest.relative_to(dest.parent.parent.parent)}")
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return True


def fetch_fontshare(root: Path, *, existing_normalized: set[str], dry_run: bool) -> tuple[int, int]:
    """Returns (new_ttfs, skipped_families)."""
    print(">> Fontshare catalog")
    catalog_raw = http_get(FONTSHARE_LIST, accept="application/json")
    catalog = json.loads(catalog_raw)
    fonts = catalog.get("fonts", [])
    print(f"   total: {len(fonts)} families")

    # Sans + Sans-leaning combo categories.
    sans = [
        f for f in fonts
        if "sans" in (f.get("category") or "").lower()
    ]
    print(f"   sans (incl. multi-cat): {len(sans)}")

    new_files = 0
    skipped = 0
    for i, f in enumerate(sans):
        name = f["name"]
        slug = f["slug"]
        norm = normalize(name)
        if norm in existing_normalized:
            print(f"  [{i+1}/{len(sans)}] skip (dup): {name}")
            skipped += 1
            continue

        fam_dir = root / family_dir_name(name)
        # Skip if we've already pulled this family.
        if fam_dir.exists() and any(fam_dir.glob("*.ttf")):
            print(f"  [{i+1}/{len(sans)}] skip (have): {name}")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(sans)}] {name}  → {fam_dir.name}/")
        if dry_run:
            new_files += 1  # count families that would be downloaded
            continue
        time.sleep(POLITE_DELAY)
        try:
            zip_bytes = http_get(FONTSHARE_DOWNLOAD.format(slug=slug))
        except urllib.error.HTTPError as e:
            print(f"      HTTP {e.code} on download, skipping", file=sys.stderr)
            continue
        except Exception as e:
            print(f"      fetch failed: {e}", file=sys.stderr)
            continue

        added = 0
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.namelist():
                    if not member.lower().endswith(".ttf"):
                        continue
                    if "/Fonts/TTF/" not in member and not member.endswith(".ttf"):
                        continue
                    stem = Path(member).stem
                    # Variable fonts (e.g. "Switzer-Variable.ttf") aren't a
                    # single weight — skip them; we want one TTF per weight.
                    if "variable" in stem.lower() or "italic" in stem.lower():
                        continue
                    weight = _fontshare_weight_for(f, stem)
                    if weight is None or weight not in KEEP_WEIGHTS:
                        continue
                    body = zf.read(member)
                    if write_ttf(fam_dir, slug, weight, body, dry_run=dry_run):
                        added += 1
        except zipfile.BadZipFile:
            print(f"      bad zip, skipping", file=sys.stderr)
            continue

        print(f"      +{added} TTFs")
        new_files += added

    return new_files, skipped


def _fontshare_weight_for(font_obj: dict, stem: str) -> int | None:
    """Match a TTF filename back to its style weight via the API metadata."""
    # The API gives us style.weight.number. Filenames look like
    # "GeneralSans-Bold.ttf", "GeneralSans-Light.ttf", etc. We can map either
    # by parsing the stem against known labels, or by matching style names.
    suffix = stem.split("-")[-1].lower() if "-" in stem else stem.lower()
    suffix = suffix.replace(" ", "")
    if suffix in LOTM_WEIGHT_NAMES:
        return LOTM_WEIGHT_NAMES[suffix]
    # Fallback: scan styles for one whose label matches the suffix.
    for style in font_obj.get("styles", []):
        if style.get("is_italic") or style.get("is_variable"):
            continue
        label = (style.get("weight", {}).get("label") or "").lower().replace(" ", "")
        if label and label in suffix:
            return int(style["weight"]["number"])
    return None


def fetch_lotm(root: Path, *, existing_normalized: set[str], dry_run: bool) -> tuple[int, int]:
    print(">> League of Moveable Type (GitHub org: theleagueof)")
    # Sans-leaning + geometric repos from the org list. Skip explicit non-sans
    # (sorts-mill-goudy, fanwood, linden-hill, prociono, etc.) but keep the
    # display sans-serif families.
    SANS_LEANING = {
        "league-gothic", "junction", "sniglet", "blackout", "orbitron",
        "raleway", "ostrich-sans", "knewave", "league-spartan",
        "league-mono", "the-neue-black",
    }
    repos_raw = http_get(f"{GITHUB_API}/orgs/{LOTM_ORG}/repos?per_page=100",
                         accept="application/vnd.github+json")
    repos = json.loads(repos_raw)
    sans_repos = [r for r in repos if r["name"] in SANS_LEANING]
    print(f"   sans-leaning repos: {len(sans_repos)}")

    new_files = 0
    skipped = 0
    for i, repo in enumerate(sans_repos):
        repo_name = repo["name"]
        # Family display name: title-case slug
        display = " ".join(p.title() for p in repo_name.split("-"))
        norm = normalize(display)
        if norm in existing_normalized:
            print(f"  [{i+1}/{len(sans_repos)}] skip (dup): {display}")
            skipped += 1
            continue

        fam_dir = root / family_dir_name(display)
        if fam_dir.exists() and any(fam_dir.glob("*.ttf")):
            print(f"  [{i+1}/{len(sans_repos)}] skip (have): {display}")
            skipped += 1
            continue

        print(f"  [{i+1}/{len(sans_repos)}] {display}  → {fam_dir.name}/")
        if dry_run:
            new_files += 1
            continue
        time.sleep(POLITE_DELAY)
        # Find latest release zip; fall back to tarball.
        try:
            rel_raw = http_get(
                f"{GITHUB_API}/repos/{LOTM_ORG}/{repo_name}/releases/latest",
                accept="application/vnd.github+json",
            )
            rel = json.loads(rel_raw)
            assets = rel.get("assets", [])
            zip_asset = next(
                (a for a in assets if a["name"].lower().endswith(".zip")),
                None,
            )
            archive_url = zip_asset["browser_download_url"] if zip_asset else None
        except urllib.error.HTTPError as e:
            if e.code == 404:
                archive_url = None
            else:
                print(f"      release HTTP {e.code}, skipping", file=sys.stderr)
                continue

        if archive_url is None:
            # Fall back to repo tarball (main branch).
            archive_url = f"https://codeload.github.com/{LOTM_ORG}/{repo_name}/zip/refs/heads/master"

        time.sleep(POLITE_DELAY)
        try:
            zip_bytes = http_get(archive_url)
        except urllib.error.HTTPError as e:
            # Some repos use 'main' instead of 'master'.
            if e.code == 404 and "master" in archive_url:
                alt = archive_url.replace("master", "main")
                time.sleep(POLITE_DELAY)
                try:
                    zip_bytes = http_get(alt)
                except Exception as e2:
                    print(f"      both master/main failed: {e2}", file=sys.stderr)
                    continue
            else:
                print(f"      HTTP {e.code} on {archive_url}, skipping", file=sys.stderr)
                continue
        except Exception as e:
            print(f"      fetch failed: {e}", file=sys.stderr)
            continue

        added = 0
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                for member in zf.namelist():
                    if not member.lower().endswith(".ttf"):
                        continue
                    stem = Path(member).stem
                    if "italic" in stem.lower() or "variable" in stem.lower():
                        continue
                    weight = _lotm_weight_for(stem)
                    if weight is None or weight not in KEEP_WEIGHTS:
                        continue
                    body = zf.read(member)
                    if write_ttf(fam_dir, repo_name, weight, body, dry_run=dry_run):
                        added += 1
        except zipfile.BadZipFile:
            print(f"      bad zip, skipping", file=sys.stderr)
            continue

        print(f"      +{added} TTFs")
        new_files += added

    return new_files, skipped


def _lotm_weight_for(stem: str) -> int | None:
    # "League-Spartan-Bold" / "LeagueSpartan-Bold" / "OstrichSans-Black"
    parts = re.split(r"[-_ ]+", stem)
    for p in reversed(parts):
        key = p.lower()
        if key in LOTM_WEIGHT_NAMES:
            return LOTM_WEIGHT_NAMES[key]
    # No suffix usually means Regular (single-weight families like Knewave).
    return 400


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent.parent / "fonts")
    p.add_argument("--source", choices=["all", "fontshare", "lotm"], default="all")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing = {normalize(d.name) for d in args.out.iterdir() if d.is_dir()}
    print(f"existing corpus: {len(existing)} families under {args.out}")

    total_new, total_skip = 0, 0
    if args.source in ("all", "fontshare"):
        n, s = fetch_fontshare(args.out, existing_normalized=existing, dry_run=args.dry_run)
        total_new += n; total_skip += s
    if args.source in ("all", "lotm"):
        n, s = fetch_lotm(args.out, existing_normalized=existing, dry_run=args.dry_run)
        total_new += n; total_skip += s

    print(f"\ndone. +{total_new} new TTFs, {total_skip} families skipped (dup/already present)")


if __name__ == "__main__":
    main()
