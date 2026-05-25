"""Pack fonts/ → fonts_zipped/. Each font is gzipped and named by a content
hash so the repo stays compact and diffs cleanly across machines.

Run after adding new fonts to fonts/:
    .venv/bin/python scripts/pack_fonts.py
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "fonts"
DST = Path(__file__).resolve().parent.parent / "fonts_zipped"
WEIGHT_RE = re.compile(r"-(\d{3})\.(?:ttf|otf)$", re.IGNORECASE)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"source {SRC} not found")
    DST.mkdir(exist_ok=True)
    manifest: dict[str, dict] = {}
    n_new = 0
    n_total = 0
    for path in sorted(list(SRC.rglob("*.ttf")) + list(SRC.rglob("*.otf"))):
        n_total += 1
        family = path.parent.name
        m = WEIGHT_RE.search(path.name)
        weight = int(m.group(1)) if m else 400
        content = path.read_bytes()
        sha16 = hashlib.sha256(content).hexdigest()[:16]
        out_path = DST / f"{sha16}.bin"
        if not out_path.exists():
            out_path.write_bytes(gzip.compress(content, compresslevel=9))
            n_new += 1
        manifest[sha16] = {
            "family": family,
            "weight": weight,
            "basename": path.name,
            "ext": path.suffix.lower().lstrip("."),
        }

    manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    (DST / ".manifest").write_text(base64.b64encode(manifest_json).decode())

    valid = {f"{k}.bin" for k in manifest} | {".manifest"}
    stale = 0
    for p in DST.iterdir():
        if p.name not in valid:
            p.unlink()
            stale += 1

    print(f"packed {n_total} fonts → {DST}")
    print(f"  new: {n_new}  stale removed: {stale}  total entries: {len(manifest)}")


if __name__ == "__main__":
    main()
