"""Extract geometric anchors from the input raster (e.g. badge center).

For text-on-arc placement, knowing the badge's center matters: both
"CAMP TINKER" and "2025" sit on arcs that are concentric with the
double ring. Finding that center first lets the optimizer search
along radii rather than over arbitrary (center_x, center_y, radius)
combinations.
"""
from __future__ import annotations

import cv2
import numpy as np


def find_badge_center(img_bgr: np.ndarray) -> tuple[float, float, float] | None:
    """Return (cx, cy, r) of the dominant ring in the image, or None.

    Uses cv2.HoughCircles on the grayscale, with the search range scaled to
    the image size. Picks the largest detected circle (a badge's outer
    ring is typically the strongest signal).
    """
    if img_bgr.ndim == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    H, W = gray.shape[:2]
    short = min(H, W)
    # Blur a touch to suppress edge noise
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.0,
        minDist=int(short * 0.4),
        param1=120, param2=60,
        minRadius=int(short * 0.30),
        maxRadius=int(short * 0.45),
    )
    if circles is None:
        return None
    found = circles[0]
    # Prefer the circle with the largest radius (the outer ring of a badge
    # is usually the strongest and largest signal).
    largest = max(found, key=lambda c: c[2])
    return float(largest[0]), float(largest[1]), float(largest[2])
