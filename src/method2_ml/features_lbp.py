"""Ekstrakcja cech LBP (Local Binary Patterns).

LBP opisuje lokalną teksturę obrazu poprzez porównanie jasności każdego piksela
z jego sąsiadami i kodowanie wyników jako binarny wzorzec → histogram.

Dla biometrii ust: dobrze rozróżnia różne wzorce tekstury bruzd.
"""

from __future__ import annotations

import numpy as np
from skimage.feature import local_binary_pattern


def extract_lbp(
    gray_image: np.ndarray,
    n_points: int = 8,
    radius: int = 1,
    method: str = "uniform",
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Oblicz cechy LBP z obrazu w skali szarości.

    Args:
        gray_image: obraz grayscale (2D uint8).
        n_points: liczba punktów sąsiedztwa (P w LBP_P,R).
        radius: promień sąsiedztwa (R w LBP_P,R).
        method: 'uniform' (zalecane — odporne na rotację), 'default', 'ror', 'nri_uniform'.
        n_bins: liczba binów histogramu. Dla method='uniform' zazwyczaj P+2.

    Returns:
        Krotka (lbp_image, histogram):
            lbp_image — mapa wzorców LBP (do wizualizacji),
            histogram — znormalizowany wektor cech.
    """
    lbp_image = local_binary_pattern(gray_image, P=n_points, R=radius, method=method)

    # Histogram wzorców LBP (znormalizowany)
    hist, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype(np.float32)

    return lbp_image, hist


def lbp_feature_vector(
    gray_image: np.ndarray,
    n_points: int = 8,
    radius: int = 1,
    method: str = "uniform",
    n_bins: int = 10,
) -> np.ndarray:
    """Zwróć tylko wektor cech LBP (bez obrazu wizualizacji)."""
    _, hist = extract_lbp(gray_image, n_points=n_points, radius=radius,
                          method=method, n_bins=n_bins)
    return hist
