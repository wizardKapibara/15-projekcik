"""Porównanie obrazów przez histogramy (korelacja histogramów jasności)."""

from __future__ import annotations

import cv2
import numpy as np


def histogram_score(img1: np.ndarray, img2: np.ndarray, bins: int = 64) -> float:
    """Oblicz podobieństwo histogramów dwóch obrazów.

    Używa korelacji histogramów (cv2.HISTCMP_CORREL): wynik ∈ [-1, 1].
    Wyższy = bardziej podobne (1 = identyczne).

    Args:
        img1, img2: obrazy grayscale lub RGB.
        bins: liczba binów histogramu.

    Returns:
        Score ∈ [-1, 1].
    """
    def _gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return img

    g1 = _gray(img1).astype(np.float32)
    g2 = _gray(img2).astype(np.float32)

    h1 = cv2.calcHist([g1], [0], None, [bins], [0, 256])
    h2 = cv2.calcHist([g2], [0], None, [bins], [0, 256])

    cv2.normalize(h1, h1)
    cv2.normalize(h2, h2)

    score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
    return float(score)
