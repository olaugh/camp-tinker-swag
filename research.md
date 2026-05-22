# Phase 1: Raster→Editable‑SVG pipeline — Research report

Date: 2026-05-21
Target asset: `camp_tinker_2025.png` — black‑on‑white badge: double ring,
radial sun rays inside the upper half, three mountain silhouettes with a
snow cap on the central peak, vertical strokes + two pine trees below
the mountains, arched **CAMP TINKER** *above* the outer ring, **2025**
centered below the badge in straight geometric sans.

## 0. Executive summary

**Principle: no custom computer vision.** Every stage below is a call
into a maintained library. The only code we write is glue (data
shuttling, scoring, the grid search). There is no end-to-end library
that does "raster logo → SVG with editable `<text>` for text regions
and `<path>` for everything else" — I checked Aspose.SVG, Inkscape's
CLI trace, glyphtracy, vtracer, and the obvious commercial tools.
They all flatten text into paths or skip text entirely. So our novelty
is *which* libraries we wire together and in *what order*, not new CV.

Recommended stack (all permissively licensed, all maintained in the last 12 months):

| Stage | Pick | Why |
|---|---|---|
| Text detection (polygon) | **PaddleOCR PP‑OCRv5 detector** (`PP-OCRv5_server_det`) | DB‑based, returns polygon vertices, handles curved baselines, MIT‑style license, weights on HF, fast on MPS via ONNX. |
| OCR (recognition) | **PaddleOCR PP‑OCRv5 recognizer** | Pair with the detector, supports rectified arbitrary‑shape crops. Tesseract kept as a sanity baseline. |
| Font ID | **Render‑and‑diff over a curated Google Fonts corpus**, optionally pre‑filtered by **FontCLIP** image embeddings | The user explicitly wants pixel‑level verifiability as the ranking metric. Render → SSIM/L1 against the original crop is the ground truth ranker; FontCLIP just prunes the corpus from ~50 to ~10 candidates. |
| Baseline fitting | **scikit‑image `CircleModel` + RANSAC** on polygon centerline; fall back to `LineModelND` for straight text | Library call, deterministic with a fixed seed, no hand‑rolled CV. |
| Text removal | **Binary mask + white fill** on this image (it's B/W). Keep `simple-lama-inpainting` wired in as a `--inpaint lama` fallback for future assets. | The asset has no texture to reconstruct. LaMa is overkill here. |
| Vectorization | **vtracer (binary mode)** via the official PyO3 bindings | Active (0.6.15, Mar 2026), MIT, handles concentric strokes cleanly, produces compact paths. Potrace kept as a `--tracer potrace` fallback. |
| SVG assembly | **Direct XML** (write `lxml`) | Simpler than svgwrite for the `<defs><path id="arc"><textPath href="#arc">` pattern. Hand‑readable output. |
| Refinement loop | **CairoSVG render → skimage SSIM + L1 → grid search** | Deterministic, fast (≤ 1 s/iteration at 1024×1024). |
| Print export | **fontTools** `SVGPathPen` on the chosen font's TTF | Converts `<text>` to `<path>` for the screen printer. |

Bottom line: nothing on this target asset requires a neural tracer or a
diffusion inpainter, and we write **no original CV**. The render‑and‑diff
font identifier is the only stage without a turnkey library, but even it
is glue: `fontTools` renders the candidate, `Pillow + freetype` rasterizes
it, `scikit-image` scores SSIM, `scipy.optimize` does the grid search.
No model training, no hand-rolled tracer, no custom segmentation.

### What I considered for "do the whole thing in one library"
| Tool | Verdict |
|---|---|
| **Aspose.SVG** `vectorize_text` | Flattens text to paths — opposite of what we want. |
| **Inkscape CLI** `--actions=SelectionTrace` | Same — text becomes paths. OCR has been a "won't fix" feature request since 2007. |
| **vtracer / potrace alone** | Geometry only, no text awareness. |
| **glyphtracy** | Built for font‑glyph tracing, not full logos. |
| **Adobe Illustrator Image Trace** | Proprietary, GUI, no text preservation. |
| **VLM end-to-end** (give Claude the PNG, ask for SVG) | Tried in head: hallucinates geometry, not pixel‑accurate, fails criterion #1. Keep VLMs to font‑name suggestion only. |

So the answer to "is there a library that just does this?" is **no**,
and the cleanest path is the glue stack above.

---

## 1. Text detection with polygon output

The reference badge has two text regions:
- "CAMP TINKER": arched along an arc *outside* the outer ring,
  spanning roughly the top 130° of a circle.
- "2025": straight, centered below the badge.

Detection must return **polygon** vertices (≥ 8 points around the arc),
not just a rotated bbox, so we can fit a circular baseline.

### Candidates

| Detector | License | Polygon? | Curved‑text quality | Maintenance | Notes |
|---|---|---|---|---|---|
| **PaddleOCR PP‑OCRv5 det** | Apache‑2.0 | Yes (DB → polygon) | Good (DB + adaptive thresholds; v5 explicitly targets curved/rotated/vertical) | Active (v5 on HF) | Battle‑tested; ONNX export works on MPS. |
| **CRAFT** (clovaai) | MIT | Character‑level polygons | Excellent on CTW‑1500 / Total‑Text | Quiet since 2021 but stable; many forks | Affinity‑map style; great precision on curved text. |
| **DBNet++** (MMOCR / docTR) | Apache‑2.0 | Yes | Best F1 on FUNSD in recent benchmarks | Active via docTR & MMOCR | Slightly heavier than DB. |
| **docTR (Mindee)** | Apache‑2.0 | Yes (uses DBNet/DBNet++ under the hood) | Good; designed for documents | Active | Cleanest Python API of the bunch. |
| **EasyOCR** | Apache‑2.0 | Quad only (rotated bbox) | Mediocre on tight arcs | Active | Rejected — quad output isn't enough for a 130° arc. |
| **TextFuseNet** | Research code | Yes | SOTA on Total‑Text in 2020 | Stale | Rejected — maintenance risk. |

### How DB works (one paragraph)
Differentiable Binarization predicts two maps per pixel: probability of
belonging to a text region and an adaptive threshold. A learned
"approximate step" combines them so the segmentation network can be
trained end‑to‑end on the binary output. At inference, connected
components in the binarized map are unwrapped into polygon vertices
(Vatti / Clipper). For curved text, the polygon is sampled densely
along its perimeter rather than being reduced to a quad, which is what
gives us the baseline curve we need.

### Failure modes on geometric line‑art logos
1. **Outer ring strokes touch the letters at thin spots.** The detector
   may merge ring + letter into one polygon. Mitigation: morphological
   opening on the input by a 1–2 px kernel before detection, OR run
   detection on the residual after ring removal (chicken/egg; we do a
   two‑pass).
2. **Closed letterforms (O, A, P, R) get treated as separate components
   when binarization is too aggressive.** Mitigation: lower
   `db_thresh` (PaddleOCR has a knob for this) until characters merge
   into words.
3. **Background = pure white means almost any detector works.** This
   asset is much easier than typical scene text.

### Recommendation
**PaddleOCR PP‑OCRv5 detector**, with CRAFT kept available behind a
`--detector craft` flag. PaddleOCR ships the model on Hugging Face
(`PaddlePaddle/PP-OCRv5_server_det`), exports cleanly to ONNX, and gives
us polygons directly.

**Install:** `pip install paddleocr paddlepaddle` (CPU) or
`paddlepaddle-gpu` on the 5090 box. MPS via ONNX:
`pip install onnxruntime`.

---

## 2. OCR / text recognition

Given a rectified text crop, we want a string. Robustness to curved
baselines matters because the rectification will be imperfect.

### Candidates

| Recognizer | License | Curved‑text robustness | Notes |
|---|---|---|---|
| **PaddleOCR PP‑OCRv5 rec** | Apache‑2.0 | Good; trained on rectified arbitrary‑shape crops | Pairs naturally with the detector. |
| **docTR recognizer** (PARSeq / CRNN / MASTER) | Apache‑2.0 | PARSeq is SOTA on irregular text | Cleanest Python API. |
| **TrOCR (microsoft/trocr-base-printed)** | MIT | Excellent on printed text | Heavier (Transformer); overkill for two short strings. |
| **Tesseract 5** | Apache‑2.0 | Poor on curved/arched text without rectification | Keep as sanity baseline only. |
| **EasyOCR** | Apache‑2.0 | OK; less accurate than PaddleOCR on benchmarks | Skip. |

### Recommendation
**PaddleOCR recognizer** as primary, **TrOCR** as a confidence
tie‑breaker when PaddleOCR returns a confidence below 0.9. Tesseract
runs only in `--debug` mode to compare.

**Install:** included with `paddleocr`. TrOCR via `transformers`.

### How recognition handles curved text
We unwarp the polygon to a rectangle first. Specifically: sample the
polygon's top and bottom edges as parametric curves, fit a circular arc
through the centerline, then resample the crop along the arc into a
straight rectangle (a polar unwrap when the arc is part of a circle).
This is a 20‑line OpenCV routine; both PaddleOCR and docTR do something
similar internally but exposing it ourselves keeps the geometry
recoverable for the SVG `<textPath>` later.

---

## 3. Font identification

**This is the hardest piece** and the user explicitly wants pixel‑level
verifiability as the ranking metric. The plan separates "narrow the
field" (a fast retrieval step) from "pick the winner" (a slow
render‑and‑diff step). Only the latter decides; the former can be
wrong without ruining accuracy.

### Candidates

| Approach | License / cost | Pros | Cons |
|---|---|---|---|
| **DeepFont (Adobe, 2015)** + community reimplementations | MIT (reimpls) | Established baseline; AdobeVFR has 2,383 categories | Trained on commercial fonts not in our Google Fonts corpus; ~80% top‑5 on its own dataset; no recent maintenance. |
| **Adobe Fonts WhatTheFont API** | Commercial, proprietary | ~85–92% top‑5 accuracy reported | Violates "no proprietary APIs in default path" constraint; not reproducible. |
| **HF `font-detector` models** | Mixed (check per‑model) | Easy to drop in | Small training sets; reportedly weak on geometric sans where many fonts look near‑identical (Poppins vs Nunito vs Quicksand). |
| **CLIP / FontCLIP image retrieval** over a rasterized corpus | MIT (CLIP), research‑license (FontCLIP) | Fast (10s of ms per query after embedding cache); great for narrowing | FontCLIP was trained for *semantic* font retrieval ("formal", "playful"), not visual identification — papers (e.g. *Texture or Semantics? VLMs Get Lost in Font Recognition*, 2025) show VLMs/CLIP get only ~50% top‑10 on hard fonts. **Use only as a pruner.** |
| **Fine‑tuned ViT classifier over our corpus** | MIT | Can hit >95% on a small closed set | Requires training; overkill for a 15‑font corpus. |
| **Render‑and‑diff** (rasterize each candidate at detected size, SSIM/L1 against the source crop) | Free, deterministic | Pixel‑level verifiable; ranking metric is the thing the user cares about; works on any closed corpus | Slower (O(N_fonts · N_weights · N_offsets) — ~10s per text region at our scale). |
| **VLM zero‑shot (Claude/GPT‑4V/Qwen2‑VL)** | Paid API or local GPU | Sometimes surprisingly good at narrowing | Not reproducible; the 2025 *Texture or Semantics?* paper shows VLMs lean on semantic cues and miss visual ones; keep behind `--vlm-assist` flag only. |

### Recommended pipeline

```
detected_crop ──► [optional FontCLIP] ──► top‑K candidates (K≈10)
                                            │
                                            ▼
                                       for each (font, weight) in K:
                                         render string at detected size
                                         ↓ try (sx, sy, letter‑spacing, weight) grid
                                         compute SSIM + L1 vs crop
                                            │
                                            ▼
                                       winner = argmin(weighted L1 + (1-SSIM))
```

The render‑and‑diff stage **is the ranker**, not just a confidence
score. This satisfies success criterion #3 (SSIM ≥ 0.97 on isolated
text).

### Corpus

Default bundled corpus (Phase 2 deliverable: ship the .ttf files):

Geometric sans, multiple weights (Regular, Medium, SemiBold, Bold,
ExtraBold, Black where available):

- Montserrat
- Poppins
- Nunito
- Raleway
- Quicksand
- Work Sans
- Inter
- Manrope
- Lato
- Open Sans
- Oswald (condensed)
- Bebas Neue (condensed display)
- League Spartan
- Archivo
- Barlow

That's ~15 families × ~5 weights = ~75 candidates. The render‑and‑diff
loop, parallelized, hits well under 60 s total.

### My prior on `camp_tinker_2025.png`
Eyeballing the badge, the letters look like a **bold geometric sans**
with closed apertures and a high crossbar on the A. Top suspects before
running the pipeline: **Montserrat ExtraBold / Black**, **Poppins
ExtraBold / Black**, **Raleway Black**, **Nunito Black**. The pipeline
should pick one of these; if it picks something else, the render‑and‑diff
score is what we trust.

### How render‑and‑diff works (concretely)
1. Render the candidate font with the detected string at the detected
   x‑height using Pillow + `freetype-py`.
2. Geometrically align: solve for the best (scale, translation,
   letter‑spacing) via golden‑section search on the L1 of the binary
   masks.
3. Score: `0.5 * (1 - SSIM(rendered, crop)) + 0.5 * L1(rendered, crop) / 255`.
4. Optionally refine weight via interpolation — but for our discrete
   corpus, weights are discrete TTFs.

### Failure modes
1. **Sub‑pixel kerning differences swamp the L1.** Mitigation: blur
   both images by σ=0.8 before scoring.
2. **Anti‑aliasing levels differ** between Pillow's freetype rasterizer
   and whatever rendered the source. Mitigation: render at 4× and
   downsample; compare on the binary mask only when SSIM disagrees
   strongly with L1.
3. **Italic / variable‑axis fonts have a continuous space we won't
   search.** Mitigation: the corpus is intentionally non‑variable;
   document the limitation.

---

## 4. Baseline‑curve fitting

Given a polygon with 2N vertices around an arched word, recover:
- the center `(cx, cy)` of the implied circle
- its radius `r`
- start/end angles `θ0, θ1`
- the per‑character font size (≈ polygon height in normal‑to‑arc direction)

### Approach (all library calls)
1. Walk the polygon. Compute the "top edge" and "bottom edge" by sorting
   vertices by angular position around the polygon centroid. (numpy)
2. Take the midpoints of the two edges as the **centerline**.
3. Fit a circle: `skimage.measure.CircleModel().estimate(centerline)`
   or `ransac(centerline, CircleModel, ...)` for noise rejection.
   Closed form, fast.
4. Angles: take the angular position of the first and last vertex.
5. Font size: average normal‑direction polygon thickness (numpy).
6. If the circle residual exceeds the line residual,
   `LineModelND().estimate(centerline)` instead — same scikit‑image API.

### Straight text
If the residual after circle fitting is greater than the line‑fit
residual, switch to a linear baseline. For "2025" this happens
automatically.

### Failure modes
- A 130° arc gives plenty of curvature signal; a 20° arc could be
  fit either way. Use the residual ratio as the switch, with a
  tiebreaker of "if center is well inside the badge, prefer circle".
- The polygon may have noise from antialiased edges. Smooth via
  Chaikin once before fitting.

### Recommendation
**scikit-image `CircleModel` / `LineModelND` + RANSAC**. No custom CV.

---

## 5. Text removal / inpainting

For `camp_tinker_2025.png` (B/W, white background), inpainting reduces
to **rasterizing the detected polygons onto a mask and painting the
mask white**. There is no texture to reconstruct.

### Strategy
1. From step 1 we have text polygons. Inflate by 2–3 px so we cover
   anti‑aliasing fringes.
2. `cv2.fillPoly(image, polygons, color=(255,255,255))`.
3. Done.

### When to use LaMa
If we ever process a colored or textured logo, drop in
`simple-lama-inpainting` behind a `--inpaint lama` flag. The library
takes `(image, binary_mask_255)` and returns a PIL image. It has a
working PyPI package as of 2026 and runs CPU or CUDA. On MPS it falls
back to CPU but is still ~1 s on a 1024² image, which is acceptable.

### Failure modes
1. Text touches the ring (likely for this asset because "CAMP TINKER"
   is right above the outer ring, but with a clear gap — visually
   confirmed). If a stroke crosses, white‑fill will cut a notch in the
   ring. Mitigation: after fill, restore the original ring by
   re‑rendering it from the *traced* SVG and OR‑ing it back. (We have
   the SVG before we composite text.)

### Recommendation
**White fill** as default; LaMa behind a flag.

---

## 6. Vectorization of the residual line‑art

Once text is removed we have: two concentric circles, ~50 radial rays,
a semicircular sun arc, three mountain polylines with a snow cap on
the central peak, vertical lines under the mountains, two pine trees.
All are crisp black strokes on white. **This is the ideal case for
classical bilevel tracers.**

### Candidates

| Tracer | License | Lang | Strengths | Weaknesses |
|---|---|---|---|---|
| **vtracer** | MIT | Rust + PyO3 | Multi‑color and binary modes; designed for line art; modern; tunable curve fitting | Binary mode optimized for thick shapes — may over‑smooth thin rays. |
| **potrace** (Selinger) | GPL | C | Gold standard for B/W; clean Bezier output | GPL (consider for the optional flag; **do not** make the default if downstream redistribution matters). Python bindings (`pypotrace`) have notoriously fragile installs. |
| **tatarize/potrace** (pure Python port) | GPL | Python | No native deps | ~500× slower; still GPL. |
| **autotrace** | GPL | C | Good corner detection | GPL; less active. |
| **DiffVG / LIVE / Im2Vec** | Research | Python + CUDA | Differentiable, can match the input image via gradient descent | Massive overkill, slow, requires CUDA, harder to make deterministic, output paths are not human‑readable. Rejected. |

### Recommendation
**vtracer in binary mode** as default. License is MIT (compatible with
anything), package is actively released (`vtracer 0.6.15` on PyPI as of
March 2026), it has a stable PyO3 Python API, and its curve‑fitting
parameters (`corner_threshold`, `length_threshold`, `splice_threshold`)
give us enough knobs to clean up the rays without hand‑tuning per
asset.

Keep **potrace** behind `--tracer potrace` for cases where vtracer
over‑smooths thin rays; on the test asset I expect the rays to be 1–2
px wide, where vtracer sometimes drops detail. Use the `pypotrace` C
bindings only on Linux (install is painful on Mac); fall back to the
pure‑python port on macOS.

### vtracer config plan for this asset

```python
vtracer.convert_image_to_svg_py(
    input_path=residual_png,
    output_path=traced_svg,
    colormode="binary",
    hierarchical="stacked",
    mode="spline",            # Bezier output
    filter_speckle=4,         # drop sub-4px noise
    corner_threshold=60,      # detect mountain peaks as corners
    length_threshold=4.0,
    splice_threshold=45,
    path_precision=2,
)
```

### Failure modes
1. **Concentric rings traced as two paths or as one path with a hole**
   depending on `hierarchical` setting. We want two separate paths;
   `hierarchical="stacked"` gives that.
2. **Rays bleed together** at the center if their tips touch. Visual
   inspection of the asset suggests there's a small clear margin —
   should be fine.
3. **Anti‑alias halos** show up as gray pixels and create extra
   contours. Binarize the input first (Otsu) before passing to vtracer.

---

## 7. SVG assembly

Final document layout:

```xml
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 W H">
  <defs>
    <path id="camp-tinker-arc" d="M ... A r r 0 0 1 ..."/>
  </defs>

  <!-- traced geometry (everything except text) -->
  <g id="art">
    <path d="..."/>
    <!-- … -->
  </g>

  <!-- editable text -->
  <text id="camp-tinker" font-family="Montserrat" font-weight="800"
        font-size="..." letter-spacing="..." text-anchor="middle">
    <textPath href="#camp-tinker-arc" startOffset="50%">CAMP TINKER</textPath>
  </text>

  <text id="year" x="..." y="..." font-family="Montserrat"
        font-weight="800" font-size="..." text-anchor="middle">2025</text>
</svg>
```

### Library choice
- **svgwrite** is fine but verbose for the `<textPath>` pattern and
  doesn't produce the cleanest output.
- **lxml** (direct XML construction) gives us full control and
  hand‑readable output. Recommend this.
- **cairosvg** for rasterizing the SVG back to PNG during the
  refinement loop. Stable, deterministic, MIT‑equivalent
  (LGPL — we only call it, no static linking concerns).

### Recommendation
**lxml** to build, **cairosvg** to rasterize for diffs, **rsvg-convert**
(via subprocess) as a fallback renderer if cairosvg disagrees with
browser rendering on `textPath` startOffset behavior (it occasionally
does, depending on cairosvg version).

---

## 8. End-to-end pipeline diagram

```
input.png
  │
  ├──► (1) Binarize (Otsu) ──► binary.png
  │
  ├──► (2) Detect text polygons (PaddleOCR det)
  │        ──► [poly_arch, poly_year]
  │
  ├──► (3) For each polygon:
  │        ├── unwarp crop  ──► (4) OCR (PaddleOCR rec) ──► string
  │        ├── fit baseline (circle | line)              ──► (cx,cy,r,θ0,θ1) | (x0,y0,θ)
  │        └── crop tight region                         ──► text_crop.png
  │
  ├──► (5) Font ID:
  │        for crop in [arch, year]:
  │          ├── (optional) FontCLIP top-K candidates
  │          └── render-and-diff against corpus  ──► (font, weight, size, ls, dx, dy)
  │
  ├──► (6) Inpaint:
  │        fillPoly(image, polygons, white)  ──► residual.png
  │
  ├──► (7) Vectorize residual (vtracer binary) ──► geometry.svg
  │
  ├──► (8) Assemble SVG:
  │        merge geometry + <defs><path id=arc> + <text><textPath>
  │        ──► output.svg
  │
  ├──► (9) Refinement loop:
  │        for i in range(MAX_ITERS):
  │          render(output.svg) → output.png
  │          diff = compare(output.png, input.png) restricted to text-mask
  │          if SSIM ≥ 0.98 and L1 ≤ 2/255: break
  │          adjust (size, ls, dx, dy, r) via local grid search
  │
  └──► (10) Mode dispatch:
           --mode editable: output.svg as-is
           --mode print:    convert <text> to <path> via fontTools SVGPathPen
```

---

## 9. Risk assessment for the reference asset

| Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|
| PaddleOCR merges "CAMP" + outer ring into one polygon | Medium | High | Two-pass: trace once, subtract ring stroke from the binary, re-detect. Or use a generous morphological gap. |
| Font corpus doesn't contain the actual font | Medium | Medium | If best-fit SSIM < 0.95, surface a warning in the report and fall back to outlining the original raster (no editable text, but pixel-accurate). |
| The "2025" digits are recognized as "ZOZS" or similar | Low | Low | Run TrOCR as tie-breaker; restrict the year regex to `\d{4}`. |
| vtracer over-smooths the radial rays | Medium | Low | Tunable; if it fails, fall back to potrace. |
| Arc fit picks wrong center because of asymmetric polygon | Low | Medium | Constrain center to lie within the badge bbox (we know the badge geometry from the traced rings). |
| cairosvg renders textPath differently than browsers | Low | Medium | Verify final SVG with Inkscape/Firefox in CI; if drift, use rsvg-convert in the refinement loop. |
| Determinism breaks (NN models with non-deterministic kernels) | Medium | Low | Set torch / paddle global seeds, disable cudnn benchmark, use ONNX runtime in deterministic mode. |

---

## 10. Open questions before Phase 2

1. **The reference image isn't yet checked into the repo.** Please
   commit it to `assets/camp_tinker_2025.png` (or wherever you prefer)
   so I can pin the pipeline to a known SHA. Also confirm the target
   resolution — the image you pasted looks ~1800–2000 px wide;
   pipeline thresholds (filter_speckle, polygon inflation) scale with
   that.
2. **Font corpus delivery.** Should I (a) bundle the listed Google Fonts
   .ttf files into `fonts/` in the repo (~30–50 MB), (b) use the
   `pyfonts` library's `load_google_font()` to fetch on first run, or
   (c) expect the user to provide a path? I lean **(b)** — `pyfonts` is
   pip‑installable, license‑clean, small footprint, and lets us pin
   versions via the Google Fonts CDN.
3. **License posture of the output.** The project itself looks
   MIT‑ish given the libraries listed. Potrace is GPL; if you're OK
   with GPL contagion via subprocess, we can keep it as a fallback,
   otherwise I'll drop it and rely solely on vtracer.
4. **Mac‑first vs Linux‑first.** PaddleOCR on Apple Silicon works via
   ONNX/CoreML but is slower than on CUDA. The 60 s budget is tight on
   an M‑series laptop; I'll target it but want to confirm you're
   benchmarking on M-series or on the RTX 5090 box.
5. **VLM‑assist scope.** If we add `--vlm-assist`, which model? I'd
   default to Claude (`claude-opus-4-7` or `claude-sonnet-4-6`) via
   the Anthropic SDK, called once with the text crop as input and a
   short list of candidate font names. Confirm or pick another.
6. **Determinism vs speed tradeoff.** A deterministic seed‑pinned
   render‑and‑diff stage is slower than a non‑deterministic one
   (we can't pre‑embed fonts in parallel batches with random crop
   augmentation). I'll go deterministic by default to meet success
   criterion #5. Confirm.
7. **Print‑mode rendering surface.** Screen printers I've worked with
   want `<path>` text *and* paths flattened into a single fill rule
   (`fill-rule="evenodd"`, no transforms, single‑color). Confirm those
   are the constraints, or share a sample of what they accept.
8. **Tests scope.** Success criterion #1 is SSIM ≥ 0.98 + L1 ≤ 2/255.
   Is "the reference image" the only required test case for Phase 2, or
   should I synthesize a couple of additional easy badges
   (e.g., the same template with "2026", a different camp name) for
   regression?

Once you confirm 1–8 I'll proceed to Phase 2 with the stack above:
**PaddleOCR + render‑and‑diff over Google Fonts + classical arc fit +
vtracer + lxml/cairosvg + fontTools for print export.**
