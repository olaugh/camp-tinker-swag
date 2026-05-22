"""OCR recognition adapters.

Each adapter takes a crop (an RGB ndarray of just the text region) and returns
(text, confidence).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


Recognizer = Callable[[np.ndarray], tuple[str, float]]


def recognize_tesseract(crop_rgb: np.ndarray) -> tuple[str, float]:
    import pytesseract
    from PIL import Image
    pil = Image.fromarray(crop_rgb)
    # psm 7 = treat the image as a single text line
    text = pytesseract.image_to_string(pil, config="--psm 7").strip()
    # Tesseract doesn't expose a clean confidence here; use 0.8 as a placeholder.
    return text, 0.8


def recognize_easyocr(crop_rgb: np.ndarray, reader=None) -> tuple[str, float]:
    import easyocr
    if reader is None:
        reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    results = reader.readtext(crop_rgb, detail=1, paragraph=False)
    if not results:
        return "", 0.0
    # Concatenate words in left-to-right order
    results.sort(key=lambda r: r[0][0][0])  # by min-x of top-left corner
    text = " ".join(r[1] for r in results)
    conf = float(np.mean([r[2] for r in results]))
    return text, conf


def recognize_doctr(crop_rgb: np.ndarray, predictor=None) -> tuple[str, float]:
    from doctr.models import ocr_predictor
    if predictor is None:
        predictor = ocr_predictor(pretrained=True)
    out = predictor([crop_rgb])
    words = []
    confs = []
    for page in out.pages:
        for block in page.blocks:
            for line in block.lines:
                for w in line.words:
                    words.append(w.value)
                    confs.append(float(w.confidence))
    if not words:
        return "", 0.0
    return " ".join(words), float(np.mean(confs))


REGISTRY: dict[str, Recognizer] = {
    "tesseract": recognize_tesseract,
    "easyocr":   recognize_easyocr,
    "doctr":     recognize_doctr,
}
