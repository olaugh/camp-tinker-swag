"""Text-removal / inpainting adapters."""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np


Inpainter = Callable[[np.ndarray, np.ndarray], np.ndarray]


def inpaint_white(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Trivial inpaint: paint mask white. Works perfectly on B/W logos."""
    out = img_bgr.copy()
    out[mask > 0] = (255, 255, 255)
    return out


def inpaint_telea(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV's classical TELEA inpainting -- no extra deps."""
    return cv2.inpaint(img_bgr, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def inpaint_lama(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """LaMa (heavy; requires simple-lama-inpainting + torch)."""
    from simple_lama_inpainting import SimpleLama
    from PIL import Image
    sl = SimpleLama()
    rgb = Image.fromarray(img_bgr[:, :, ::-1])
    m = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    out = np.array(sl(rgb, m))
    return out[:, :, ::-1].copy()  # RGB->BGR


REGISTRY: dict[str, Inpainter] = {
    "white": inpaint_white,
    "telea": inpaint_telea,
    "lama":  inpaint_lama,
}


def polygons_to_mask(polygons: list[np.ndarray], shape: tuple[int, int], inflate_px: int = 3) -> np.ndarray:
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
    if inflate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*inflate_px+1, 2*inflate_px+1))
        mask = cv2.dilate(mask, k)
    return mask
