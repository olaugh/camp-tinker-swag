"""Diagnose circle detection on a single image.

For each circle HoughCircles proposes, this:
  * walks 360 points around the perimeter,
  * counts how many land on ink (within +/- stroke_width/2 of the nominal r),
  * sub-pixel-refines (cx, cy, r) via skimage RANSAC + CircleModel on the
    canny edges near the nominal radius,
  * estimates the stroke width via radial probe,
  * draws every candidate on an overlay PNG (kept candidates green, rejected red).

Use it to figure out the right perimeter-coverage threshold and to confirm
the refined circle parameters before wiring them back into circles.detect.

Usage:
    DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib \\
    .venv/bin/python eval/circle_audit.py \\
        --input assets/camp_tinker_2025.png \\
        --out runs_real/circle_audit/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make sure src/ is on sys.path so we can import ctswag.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
from skimage.measure import CircleModel, ransac


def perimeter_coverage(gray: np.ndarray, cx: float, cy: float, r: float,
                       *, stroke: int = 6, n_samples: int = 720,
                       ink_threshold: int = 128) -> float:
    """Return the fraction of perimeter samples that hit ink.

    For each of n_samples angles we probe a small radial band of width
    `stroke` (pixels) centered on the nominal radius. A hit = any pixel in
    that band has gray < ink_threshold.
    """
    H, W = gray.shape[:2]
    half = max(stroke // 2, 1)
    hits = 0
    counted = 0
    for i in range(n_samples):
        theta = 2 * np.pi * i / n_samples
        dx, dy = np.cos(theta), np.sin(theta)
        band_hit = False
        for k in range(-half, half + 1):
            px = int(round(cx + (r + k) * dx))
            py = int(round(cy + (r + k) * dy))
            if not (0 <= px < W and 0 <= py < H):
                continue
            if gray[py, px] < ink_threshold:
                band_hit = True
                break
        # Only count angles where at least one sample fell in-bounds.
        if 0 <= int(round(cx + r * dx)) < W and 0 <= int(round(cy + r * dy)) < H:
            counted += 1
            if band_hit:
                hits += 1
    return hits / max(counted, 1)


def refine_circle(gray: np.ndarray, cx: float, cy: float, r: float,
                  *, band: int = 12) -> tuple[float, float, float, int]:
    """Sub-pixel refine via Canny edges + RANSAC + CircleModel.

    Returns (cx_refined, cy_refined, r_refined, n_inliers). If refinement
    fails, returns the input plus 0 inliers.
    """
    H, W = gray.shape[:2]
    # Mask annulus around the nominal ring so we only fit nearby edge pixels.
    yy, xx = np.mgrid[0:H, 0:W]
    dist = np.hypot(xx - cx, yy - cy)
    annulus = (dist >= r - band) & (dist <= r + band)

    edges = cv2.Canny(gray, 80, 160)
    pts_y, pts_x = np.where(edges & annulus)
    if pts_x.size < 50:
        return cx, cy, r, 0
    data = np.column_stack([pts_x, pts_y]).astype(np.float64)
    try:
        model, inliers = ransac(data, CircleModel,
                                min_samples=3, residual_threshold=1.5,
                                max_trials=200, random_state=0)
    except Exception:
        return cx, cy, r, 0
    if model is None or inliers is None:
        return cx, cy, r, 0
    cx_r, cy_r, r_r = model.params
    return float(cx_r), float(cy_r), float(r_r), int(inliers.sum())


def estimate_stroke(gray: np.ndarray, cx: float, cy: float, r: float,
                    *, samples: int = 36, max_probe: int = 40,
                    ink_threshold: int = 128) -> float:
    """Median radial thickness measured at sample angles. NaN if no hits."""
    H, W = gray.shape[:2]
    widths: list[int] = []
    for i in range(samples):
        theta = 2 * np.pi * i / samples
        dx, dy = np.cos(theta), np.sin(theta)
        rx = int(round(cx + r * dx))
        ry = int(round(cy + r * dy))
        if not (0 <= rx < W and 0 <= ry < H) or gray[ry, rx] >= ink_threshold:
            continue
        inner = 0
        for k in range(1, max_probe):
            qx = int(round(cx + (r - k) * dx))
            qy = int(round(cy + (r - k) * dy))
            if not (0 <= qx < W and 0 <= qy < H) or gray[qy, qx] >= ink_threshold:
                inner = k
                break
        outer = 0
        for k in range(1, max_probe):
            qx = int(round(cx + (r + k) * dx))
            qy = int(round(cy + (r + k) * dy))
            if not (0 <= qx < W and 0 <= qy < H) or gray[qy, qx] >= ink_threshold:
                outer = k
                break
        widths.append(inner + outer)
    if not widths:
        return float("nan")
    return float(np.median(widths))


def hough_candidates(gray: np.ndarray, *, param2: int = 50,
                     min_r_ratio: float = 0.20, max_r_ratio: float = 0.48
                     ) -> list[tuple[float, float, float]]:
    H, W = gray.shape[:2]
    short = min(H, W)
    blur = cv2.medianBlur(gray, 5)
    out = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=max(int(short * 0.02), 8),
        param1=120, param2=param2,
        minRadius=int(short * min_r_ratio),
        maxRadius=int(short * max_r_ratio),
    )
    if out is None:
        return []
    return [(float(c[0]), float(c[1]), float(c[2])) for c in out[0]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--coverage-threshold", type=float, default=0.75,
                    help="Min perimeter ink coverage to keep a candidate.")
    ap.add_argument("--param2", type=int, default=50)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(args.input), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"can't read {args.input}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape

    candidates = hough_candidates(gray, param2=args.param2)
    print(f"hough returned {len(candidates)} candidates (param2={args.param2})")

    rows: list[dict] = []
    for cx, cy, r in candidates:
        stroke = estimate_stroke(gray, cx, cy, r)
        cov = perimeter_coverage(gray, cx, cy, r,
                                 stroke=int(stroke) if not np.isnan(stroke) else 6)
        rcx, rcy, rr, n_in = refine_circle(gray, cx, cy, r)
        rcov = perimeter_coverage(gray, rcx, rcy, rr,
                                  stroke=int(stroke) if not np.isnan(stroke) else 6)
        rows.append({
            "hough": {"cx": cx, "cy": cy, "r": r},
            "stroke_px": stroke,
            "coverage": cov,
            "refined": {"cx": rcx, "cy": rcy, "r": rr, "inliers": n_in,
                        "coverage": rcov},
            "kept": cov >= args.coverage_threshold,
        })

    # Cluster kept candidates by refined radius and keep one per cluster.
    kept = [r for r in rows if r["kept"]]
    kept.sort(key=lambda x: -x["refined"]["r"])
    final: list[dict] = []
    r_tol = max(min(H, W) * 0.015, 4.0)
    for c in kept:
        if any(abs(c["refined"]["r"] - f["refined"]["r"]) < r_tol for f in final):
            continue
        final.append(c)

    # ---- Concentric scan ---------------------------------------------------
    # Anchor on the highest-coverage hough hit, then sweep radius only.
    # This is how we recover ALL rings that share a common center without
    # Hough's per-circle center wobble producing false positives.
    #
    # The trick: use a 1-pixel band (not stroke-wide) so adjacent radii inside
    # a single ring don't all report 1.0. The resulting curve has a plateau
    # at each ring's *width* with sharp edges at the inner/outer edge of the
    # stroke. We then find connected regions of high coverage and report each
    # region's midpoint as the ring's central radius, plus its width as the
    # ring's stroke.
    rows_sorted = sorted(rows, key=lambda r: -r["coverage"])
    anchor = rows_sorted[0] if rows_sorted else None
    concentric_peaks: list[dict] = []
    if anchor:
        acx, acy = anchor["hough"]["cx"], anchor["hough"]["cy"]
        short = min(H, W)
        r_lo, r_hi = int(short * 0.10), int(short * 0.48)
        radii = np.arange(r_lo, r_hi + 1, 1.0)
        # band=1 means "exactly at this radius" -- the curve becomes a binary
        # mask of "is there ink along this circle".
        cov_curve = np.array([
            perimeter_coverage(gray, acx, acy, float(r), stroke=1)
            for r in radii
        ])
        # Find runs of radii whose coverage >= threshold. Each run is a ring.
        above = cov_curve >= args.coverage_threshold
        i = 0
        while i < len(above):
            if not above[i]:
                i += 1
                continue
            j = i
            while j < len(above) and above[j]:
                j += 1
            # Run [i, j) -- accept it as a ring if it's at least 3 px wide
            # (filters out single-pixel noise from anti-aliased edges).
            if j - i >= 3:
                mid_idx = (i + j - 1) // 2
                concentric_peaks.append({
                    "cx": float(acx), "cy": float(acy),
                    "r": float(radii[mid_idx]),
                    "r_inner": float(radii[i]),
                    "r_outer": float(radii[j - 1]),
                    "stroke_px": float(radii[j - 1] - radii[i] + 1),
                    "coverage": float(cov_curve[mid_idx]),
                })
            i = j
        # Save the coverage curve so we can plot/sanity-check later.
        (args.out / "concentric_curve.json").write_text(json.dumps({
            "anchor": {"cx": acx, "cy": acy},
            "radii": radii.tolist(),
            "coverage": cov_curve.tolist(),
        }))

    # Print human report
    print(f"\n{'KEEP':<5} {'cov':>6} {'stroke':>7} {'cx_h':>7} {'cy_h':>7} {'r_h':>7}  "
          f"-> {'cx_r':>7} {'cy_r':>7} {'r_r':>7} {'cov_r':>6} {'inl':>5}")
    for r in rows:
        mark = "yes" if r["kept"] else "."
        rf = r["refined"]
        print(f"{mark:<5} {r['coverage']:>6.3f} {r['stroke_px']:>7.2f} "
              f"{r['hough']['cx']:>7.1f} {r['hough']['cy']:>7.1f} {r['hough']['r']:>7.1f}  "
              f"-> {rf['cx']:>7.1f} {rf['cy']:>7.1f} {rf['r']:>7.1f} "
              f"{rf['coverage']:>6.3f} {rf['inliers']:>5}")
    print(f"\nfinal kept after radius-dedup: {len(final)}")
    for f in final:
        rf = f["refined"]
        print(f"  cx={rf['cx']:.2f} cy={rf['cy']:.2f} r={rf['r']:.2f} "
              f"stroke={f['stroke_px']:.2f} coverage={rf['coverage']:.3f}")

    print(f"\nconcentric scan around anchor "
          f"({anchor['hough']['cx']:.1f}, {anchor['hough']['cy']:.1f}) "
          f"-> {len(concentric_peaks)} rings (coverage >= {args.coverage_threshold}):")
    for p in concentric_peaks:
        print(f"  cx={p['cx']:.2f} cy={p['cy']:.2f} r={p['r']:.2f} "
              f"stroke={p['stroke_px']:.2f} inner={p['r_inner']:.0f} "
              f"outer={p['r_outer']:.0f} coverage={p['coverage']:.3f}")

    # Overlay
    overlay = bgr.copy()
    for r in rows:
        rf = r["refined"]
        color = (0, 180, 0) if r["kept"] else (0, 0, 180)
        cv2.circle(overlay, (int(rf["cx"]), int(rf["cy"])), int(rf["r"]),
                   color, 2)
    for f in final:
        rf = f["refined"]
        cv2.circle(overlay, (int(rf["cx"]), int(rf["cy"])), int(rf["r"]),
                   (255, 128, 0), 3)  # hough-dedup kept = orange

    # Concentric-scan ring overlay (cyan, drawn last so they sit on top).
    concentric_only = bgr.copy()
    for p in concentric_peaks:
        cv2.circle(concentric_only, (int(p["cx"]), int(p["cy"])), int(p["r"]),
                   (200, 200, 0), 3)
        cv2.drawMarker(concentric_only, (int(p["cx"]), int(p["cy"])),
                       (0, 0, 200), markerType=cv2.MARKER_CROSS, markerSize=20,
                       thickness=2)

    cv2.imwrite(str(args.out / "overlay.png"), overlay)
    cv2.imwrite(str(args.out / "concentric.png"), concentric_only)
    (args.out / "audit.json").write_text(json.dumps({
        "input": str(args.input),
        "image_size": [W, H],
        "coverage_threshold": args.coverage_threshold,
        "all": rows,
        "final": final,
        "concentric": concentric_peaks,
    }, indent=2, default=str))
    print(f"\nwrote {args.out / 'overlay.png'} (hough dedup)")
    print(f"wrote {args.out / 'concentric.png'} (concentric scan around anchor)")


if __name__ == "__main__":
    main()
