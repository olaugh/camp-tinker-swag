# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A pipeline that turns a raster logo PNG (specifically the Camp Tinker badge) into an editable SVG: vector geometry as `<path>` + the recognized words as real `<text>` / `<textPath>` nodes, so the result is editable in Illustrator/Figma instead of being a flat tracing. The guiding principle in `research.md` is "no custom CV" — every stage is glue over a maintained library, ranked by pixel-level diff against the original raster.

## Environment setup

Python 3.11+ required (`pyproject.toml` pins `>=3.11`). The system Python on macOS is 3.9; use `uv` for a 3.13 venv:

```
uv venv --python 3.13
uv pip install --python .venv/bin/python -e ".[tesseract]"
brew install cairo tesseract
```

CairoSVG (used everywhere — synth badge, eval rendering, font-ID rasterization) loads `libcairo.2.dylib` at runtime. On macOS Homebrew the dylib is under `/opt/homebrew/lib`; export `DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib` before running anything that imports `ctswag.eval`, `ctswag.synth`, or `cairosvg`. Without it you get `OSError: no library called "cairo-2" was found`.

## Running things

Synthesize the reference badge (used as ground truth — the real reference image isn't checked in):

```
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python -m ctswag.synth --out assets --name camp_tinker_synth
```

Run the pipeline across a grid of (detector × recognizer × inpainter × tracer) combinations and write a leaderboard:

```
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \
  .venv/bin/python eval/run_combos.py \
    --input assets/camp_tinker_synth.png --out runs/ \
    --badge-synth Montserrat:800
```

Useful `run_combos.py` flags:
- `--badge-synth Family:weight` — *also* rebuilds the input PNG from `ctswag.synth` so the "synth" detector can use ground-truth geometry. Only valid when the input is synthetic.
- `--detectors`, `--recognizers`, `--tracers`, `--inpainters` — restrict the grid.
- `--limit-combos N` — first N combos only.
- `--text "CAMP TINKER" 2025` — skip OCR, assign these strings to detected polygons top-to-bottom.
- `--force-arc` — force every text region onto an arc baseline.

Fetch the font corpus (one TTF per (family, weight), required for font-ID):

```
.venv/bin/python scripts/fetch_fonts.py        # ~15 geometric sans families
.venv/bin/python scripts/fetch_fonts_full.py   # all of Google Fonts (large)
```

The corpus is `fonts/<Family>/<family>-<weight>.ttf`; `fontid.discover_corpus()` walks that tree and parses the weight from the `-NNN.ttf` suffix.

There is no test suite and no linter configured.

## Architecture

`ctswag.pipeline.run()` is the orchestrator. It threads a single image through registered adapters:

1. **detect** (`detect.REGISTRY`) → list of polygon `np.ndarray` in image coords. `synth` reads exact polygons from a `SynthBadge` (only valid for synthesized input); `tesseract`/`easyocr`/`doctr` return real detections.
2. **recognize** (`recognize.REGISTRY`) → string per polygon. For arc baselines the crop is polar-unwrapped first (`unwarp.unwarp_arc`).
3. **baseline fit** (`baseline.fit`) → `Baseline(kind="arc"|"line", params=...)` via skimage `CircleModel` / `LineModelND` + RANSAC. Per-word polygons coming from real OCR are then re-merged into multi-word arcs by `merge.maybe_merge` when their centerlines share a circle.
4. **font ID** (`fontid.match`) → top-K `FontMatch` per text via render-and-diff (PIL+freetype → composite of L1 + (1−SSIM)). The corpus is filtered by `corpus_min_weight` (e.g. 700 = bold only).
5. **optimize** (`optimize.sweep`) → coordinate-descent grid over (font from top-K, size, letter-spacing, dx, dy, candidate baselines). The optimizer can promote a `line` baseline to `arc` when arcs score better, and it uses `geom.find_badge_center` to bias arcs toward the badge's center. Without this step font-ID picks the wrong family routinely.
6. **inpaint** (`inpaint.REGISTRY`) — `white` fills text polygons with white (the asset is B/W); `telea` is OpenCV's diffusion inpainter; `lama` is the optional deep model.
7. **trace** (`trace.REGISTRY`) — `vtracer` in binary mode is the default; `potrace` is an optional fallback. Output is a `TraceResult` of raw `<path d=…>` strings.
8. **assemble** (`assemble.assemble`) — direct lxml emits `<svg>` with `<defs><path id="…-arc"/></defs>` and `<text><textPath href="…">…</textPath></text>` for arc text, plain `<text x y>` for line text. dx/dy from the optimizer are applied via `startOffset="50% + (dx/arc_len)*100%"`, NOT by mutating the arc's t0/t1, to avoid double-counting.

`PipelineConfig` controls all of this. The `refine_force_arc` flag wraps every text in an arc baseline (useful for badge layouts where all text is curved). `text_overrides` skips OCR entirely.

## Intermediate types (`types.py`)

`DetectedText` carries `polygon`, `text`, `baseline`, and `crop`. `FontMatch` carries `family`, `weight`, `size_px`, `letter_spacing_em`, `dx`, `dy`, plus the loss components (`score`, `ssim`, `l1`). `Baseline.params` is `(cx, cy, r, t0, t1)` for arcs and `(x0, y0, x1, y1)` for lines.

## Eval output layout

`eval/run_combos.py` writes per-run artifacts to `runs/<combo_name>/`: `output.svg`, `render.png` (cairosvg of the SVG), `diff.png` (abs diff vs original), `summary.json` (config + metrics + detected text + font matches). `runs/leaderboard.json` ranks all combos by (completeness desc, SSIM desc) where "completeness" is the fraction of expected words present (only computed when `--badge-synth` provides ground truth).

## Gotchas worth knowing

- The `synth` detector is a ground-truth shortcut that requires the `SynthBadge` instance — it's passed through `run(..., badge=badge)` via `detector_kwargs["badge"]`. Without it `detect_synth` raises.
- Filenames in `fonts/<Family>/` must end with `-<weight>.ttf` (e.g. `montserrat-800.ttf`). The weight regex is `-(\d{3})\.ttf$`. TTFs without that suffix default to weight 400 and may be filtered out by `corpus_min_weight`.
- `font-size ≈ cap-height / 0.72` is the conversion used in `pipeline.py` to seed the optimizer's size axis from the polygon's physical height (and minAreaRect short side for tilted line text). The optimizer then sweeps ±10% around that seed.
- Arc text in the final SVG sits on a path that extends 2× the text's angular width — this gives the `startOffset` shift room for dx adjustments without the text running off the path.
