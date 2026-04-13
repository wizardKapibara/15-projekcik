"""Porównanie obrazów metodą SSIM (Structural Similarity Index)."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def ssim_score(img1: np.ndarray, img2: np.ndarray, target_size=(256, 128)) -> float:
    """Oblicz SSIM między dwoma obrazami.

    Args:
        img1, img2: obrazy grayscale lub RGB. Zostają sprowadzone do target_size.
        target_size: (width, height) dla normalizacji rozmiaru.

    Returns:
        SSIM ∈ [-1, 1]. Wyższy = bardziej podobne.
    """
    def _prepare(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

    g1 = _prepare(img1)
    g2 = _prepare(img2)
    score, _ = structural_similarity(g1, g2, full=True)
    return float(score)
