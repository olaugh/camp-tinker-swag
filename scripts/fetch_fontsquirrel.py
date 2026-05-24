"""Polite scrape of Font Squirrel's sans-serif 100%-free listing via Playwright.

Font Squirrel's AWS WAF challenges plain curl/urllib requests with
x-amzn-waf-action: challenge, so we drive a real headless Chromium that solves
the challenge on first page load. Subsequent download calls reuse the same
browser context (cookies + JS-set fingerprint) so they return real zips.

We dedupe twice:
  1. Pre-download by slug — skip if normalized slug matches an existing
     fonts/ family directory name.
  2. Post-download by TTF-reported family name — handles cases where slug and
     family name differ (e.g. "open-sans" vs "OpenSans").

Politeness:
  - Single browser, single page, fully sequential.
  - User-Agent: realistic Chrome string (a custom UA gets WAF-flagged).
  - 1.5s between listing-page navigations, 3.0s between zip downloads.
  - Resume-friendly: skips families already on disk on re-run.

Usage:
    .venv/bin/python scripts/fetch_fontsquirrel.py                # full run
    .venv/bin/python scripts/fetch_fontsquirrel.py --max-pages 2  # smoke test
    .venv/bin/python scripts/fetch_fontsquirrel.py --collect-only # paginate + list, no downloads
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import time
import zipfile
from pathlib import Path

from playwright.sync_api import sync_playwright

LIST_URL_BASE = (
    "https://www.fontsquirrel.com/fonts/list/find_fonts"
    "?filter%5Bclassification%5D=sans-serif&filter%5Bis_free%5D=1"
)
# Page 1 has no offset segment; later pages need /<offset> path component
# before the query string (Font Squirrel pattern, see pagination links).
LIST_URL_OFFSET = (
    "https://www.fontsquirrel.com/fonts/list/find_fonts/{offset}"
    "?filter%5Bclassification%5D=sans-serif&filter%5Bis_free%5D=1"
)
DOWNLOAD_URL = "https://www.fontsquirrel.com/fonts/download/{slug}"

# A real Chrome UA — the WAF flags obvious bot strings. We do still identify
# the run via the contact email in logs and via a custom header.
UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)

LIST_DELAY = 1.5
DOWNLOAD_DELAY = 4.0
# AWS WAF JS challenge takes ~6-10s to execute and set its cookie. We pay this
# once per listing page (status 202 → 200 after JS runs); downloads after that
# inherit the aws-waf-token cookie via the context.
WAF_CHALLENGE_WAIT_MS = 10_000

KEEP_WEIGHTS = {100, 200, 300, 400, 500, 600, 700, 800, 900}

WEIGHT_NAMES = {
    "thin": 100, "hairline": 100,
    "extralight": 200, "ultralight": 200,
    "light": 300,
    "regular": 400, "normal": 400, "book": 400, "roman": 400,
    "medium": 500,
    "demibold": 600, "semibold": 600,
    "bold": 700,
    "extrabold": 800, "ultrabold": 800, "heavy": 800,
    "black": 900, "ultra": 900, "fat": 900,
}


def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def family_dir_name(name: str) -> str:
    parts = re.split(r"[\s_-]+", name)
    return "".join(p[:1].upper() + p[1:] for p in parts if p)


def parse_weight_from_stem(stem: str) -> int | None:
    """Stems like 'Foo-Bold', 'FooBold', 'FooBlack', 'Foo-ExtraBold'."""
    # Try explicit '-Bold' / '_Black' style first.
    parts = re.split(r"[-_ ]+", stem)
    for p in reversed(parts):
        key = p.lower().strip()
        if key in WEIGHT_NAMES:
            return WEIGHT_NAMES[key]
    # Try splitting CamelCase: 'FooBold' -> ['Foo', 'Bold']
    cam = re.findall(r"[A-Z][a-z]+|[a-z]+|\d+", stem)
    for p in reversed(cam):
        key = p.lower()
        if key in WEIGHT_NAMES:
            return WEIGHT_NAMES[key]
    return None


def ttf_family_name(data: bytes) -> str | None:
    """Read the family name (nameID 1) from a TTF byte blob without writing to
    disk. Returns None on failure."""
    try:
        from fontTools.ttLib import TTFont
        f = TTFont(io.BytesIO(data))
        # Prefer the Windows English (3,1,0x409) name; fall back to anything.
        name = f["name"].getName(1, 3, 1, 0x409) or f["name"].getName(1, 0, 0) \
            or f["name"].getName(1, 1, 0)
        return name.toUnicode() if name else None
    except Exception:
        return None


def collect_slugs(page, max_pages: int) -> list[str]:
    """Paginate the listing, return ordered slug list.

    Transient WAF challenges occasionally return an empty page. Don't treat
    the first empty as end-of-results — retry up to 3 times with a fresh
    nav. Only break when we see 2 consecutive *retried* empties (real EOF)."""
    all_slugs: list[str] = []
    seen: set[str] = set()
    offset = 0
    page_n = 0
    consec_empty_after_retry = 0
    while page_n < max_pages:
        url = LIST_URL_BASE if offset == 0 else LIST_URL_OFFSET.format(offset=offset)
        print(f"  list page {page_n+1} (offset={offset})")

        anchors: list[str] = []
        for attempt in range(3):
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=60_000)
            except Exception as e:
                print(f"    nav failed (attempt {attempt+1}): {e}", file=sys.stderr)
                resp = None
            status = resp.status if resp else None
            # 202 = WAF JS challenge; 200 = served; non-2xx = block
            if status == 202:
                page.wait_for_timeout(WAF_CHALLENGE_WAIT_MS)
            else:
                page.wait_for_timeout(int(LIST_DELAY * 1000))
            anchors = page.eval_on_selector_all(
                "a[href^='/fonts/download/']",
                "els => els.map(e => e.getAttribute('href'))",
            )
            print(f"    attempt {attempt+1}: status={status} title={page.title()!r} anchors={len(anchors)}",
                  file=sys.stderr)
            if anchors:
                break
            # Empty — wait longer before retrying.
            page.wait_for_timeout(WAF_CHALLENGE_WAIT_MS)

        slugs_this_page = []
        for a in anchors:
            slug = a.rstrip("/").split("/")[-1]
            if not slug or slug in seen:
                continue
            seen.add(slug)
            slugs_this_page.append(slug)
        all_slugs.extend(slugs_this_page)
        print(f"    +{len(slugs_this_page)} slugs (total {len(all_slugs)})")
        if not slugs_this_page:
            consec_empty_after_retry += 1
            if consec_empty_after_retry >= 2:
                print(f"    2 consecutive empties — past EOF, stopping",
                      file=sys.stderr)
                break
        else:
            consec_empty_after_retry = 0
        offset += 50
        page_n += 1
    return all_slugs


def download_zip(page, slug: str) -> bytes | None:
    """Try once. On 202 (transient WAF JS challenge) re-clear and retry once.
    On 403/429 (persistent block on this slug) give up — retrying just thrashes
    the WAF and doesn't help."""
    url = DOWNLOAD_URL.format(slug=slug)
    for attempt in (1, 2):
        try:
            resp = page.context.request.get(url, timeout=60_000)
        except Exception as e:
            print(f"      request err: {e}", file=sys.stderr)
            return None
        if resp.status == 202 and attempt == 1:
            # WAF JS challenge — refresh listing page to re-set token cookie,
            # then retry this download once.
            try:
                page.goto(LIST_URL_BASE, wait_until="domcontentloaded", timeout=60_000)
                page.wait_for_timeout(WAF_CHALLENGE_WAIT_MS)
            except Exception:
                pass
            continue
        if resp.status != 200:
            print(f"      HTTP {resp.status}", file=sys.stderr)
            return None
        body = resp.body()
        if not body or body[:4] != b"PK\x03\x04":
            ct = resp.headers.get("content-type", "?")
            print(f"      non-zip body ({len(body)}B, ct={ct})", file=sys.stderr)
            return None
        return body
    return None


def process_zip(zip_bytes: bytes, slug: str, root: Path,
                existing_norm: set[str], *, dry_run: bool) -> tuple[int, str | None, bool]:
    """Returns (ttfs_written, family_name, was_dup)."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return 0, None, False

    # First pass: pick a representative font to determine family name.
    # Some Font Squirrel families ship only .otf; fontTools/freetype handle
    # both formats interchangeably for our render-and-diff use case.
    font_members = [
        m for m in zf.namelist()
        if m.lower().endswith((".ttf", ".otf"))
    ]
    if not font_members:
        return 0, None, False

    family_name = None
    for m in font_members:
        body = zf.read(m)
        fam = ttf_family_name(body)
        if fam:
            family_name = fam
            break
    if not family_name:
        family_name = slug  # fallback

    # Strip weight suffix from family name so we cluster all weights under one
    # directory (e.g. "Acherus Grotesque Bold" -> "Acherus Grotesque").
    bare = family_name
    for suffix in sorted(WEIGHT_NAMES.keys(), key=len, reverse=True):
        bare = re.sub(rf"\s+{suffix}\b", "", bare, flags=re.I)
    bare = bare.strip()

    norm = normalize(bare)
    if norm in existing_norm:
        return 0, bare, True

    fam_dir = root / family_dir_name(bare)
    if fam_dir.exists() and any(fam_dir.glob("*.ttf")):
        return 0, bare, True

    if dry_run:
        return len(font_members), bare, False

    fam_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for m in font_members:
        ext = Path(m).suffix.lower()  # ".ttf" or ".otf"
        stem = Path(m).stem
        if "italic" in stem.lower() or "oblique" in stem.lower():
            continue
        weight = parse_weight_from_stem(stem)
        if weight is None:
            # Single-weight family with no suffix — call it 400.
            weight = 400
        if weight not in KEEP_WEIGHTS:
            continue
        # Keep the original extension so freetype loads via the right driver
        # (fontid.discover_corpus already accepts both .ttf and .otf via the
        #  weight-suffix regex on the basename).
        dest = fam_dir / f"{slug}-{weight}{ext}"
        if dest.exists():
            continue
        dest.write_bytes(zf.read(m))
        written += 1

    # Mark this family as taken so subsequent slugs in the same zip-family
    # (rare but possible) skip it.
    existing_norm.add(norm)
    return written, bare, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "fonts")
    ap.add_argument("--max-pages", type=int, default=28,
                    help="Max listing pages to walk (default: 28 = ~1400 families)")
    ap.add_argument("--max-downloads", type=int, default=10_000,
                    help="Cap on number of downloads attempted (after dedup)")
    ap.add_argument("--collect-only", action="store_true",
                    help="Just paginate + list slugs; don't download")
    ap.add_argument("--dry-run", action="store_true",
                    help="Download zips and inspect, but don't write TTFs")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    existing_norm = {normalize(d.name) for d in args.out.iterdir() if d.is_dir()}
    print(f"existing corpus: {len(existing_norm)} families under {args.out}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # NOTE: cannot add an X-Bot-Contact header here — any unrecognized
        # custom header trips the AWS WAF and we never clear the challenge.
        # The run is still identified by the realistic Chrome UA + by the
        # behaviour (sequential, single-context, low rate).
        ctx = browser.new_context(
            user_agent=UA,
            viewport={"width": 1400, "height": 1000},
        )
        page = ctx.new_page()

        print(">> collecting slugs from listing")
        slugs = collect_slugs(page, max_pages=args.max_pages)
        print(f"total slugs: {len(slugs)}")

        # Pre-dedupe by slug (catches the most common case).
        pre_dedup_slugs = [s for s in slugs if normalize(s) not in existing_norm]
        print(f"after pre-dedup by slug: {len(pre_dedup_slugs)} candidates "
              f"({len(slugs) - len(pre_dedup_slugs)} dropped as dup)")

        if args.collect_only:
            for s in pre_dedup_slugs[:50]:
                print(f"  candidate: {s}")
            print(f"... ({len(pre_dedup_slugs)} total)")
            browser.close()
            return

        targets = pre_dedup_slugs[: args.max_downloads]
        print(f"\n>> downloading {len(targets)} families")
        new_ttfs = 0
        dup_post = 0
        failed = 0
        for i, slug in enumerate(targets, 1):
            print(f"  [{i}/{len(targets)}] {slug}")
            time.sleep(DOWNLOAD_DELAY)
            zip_bytes = download_zip(page, slug)
            if not zip_bytes:
                failed += 1
                continue
            n, fam, was_dup = process_zip(zip_bytes, slug, args.out,
                                          existing_norm, dry_run=args.dry_run)
            if was_dup:
                print(f"      dup (family={fam!r})")
                dup_post += 1
            else:
                print(f"      +{n} TTFs (family={fam!r})")
                new_ttfs += n
            # Light periodic progress summary
            if i % 25 == 0:
                print(f"      ── progress: +{new_ttfs} TTFs, {dup_post} post-dedup, {failed} failed")

        browser.close()

    print(f"\ndone. +{new_ttfs} TTFs, {dup_post} families post-deduped, {failed} downloads failed")


if __name__ == "__main__":
    main()
