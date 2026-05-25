"""Unpack fonts_zipped/ → fonts/ for local use.

After cloning, run this once before invoking the pipeline:
    .venv/bin/python scripts/unpack_fonts.py
The fonts/ directory is gitignored; only fonts_zipped/ ships in the repo.
"""
from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "fonts_zipped"
DST = Path(__file__).resolve().parent.parent / "fonts"


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source {SRC} not found")
    manifest_path = SRC / ".manifest"
    if not manifest_path.exists():
        raise SystemExit(f"manifest {manifest_path} not found — run pack_fonts.py first")
    manifest = json.loads(base64.b64decode(manifest_path.read_text()))
    DST.mkdir(exist_ok=True)

    n_new = 0
    n_skip = 0
    for sha16, info in manifest.items():
        src = SRC / f"{sha16}.bin"
        if not src.exists():
            print(f"  missing: {sha16}.bin (skipping)")
            continue
        out_dir = DST / info["family"]
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / info["basename"]
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        out_path.write_bytes(gzip.decompress(src.read_bytes()))
        n_new += 1

    print(f"unpacked {len(manifest)} fonts → {DST}")
    print(f"  new: {n_new}  already-present: {n_skip}")


if __name__ == "__main__":
    main()
