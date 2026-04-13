"""Ekstrakcja cech HOG (Histogram of Oriented Gradients).

HOG opisuje lokalną strukturę krawędzi poprzez histogramy kierunków gradientu
w siatce komórek. Bardzo dobry dla obrazów z wyraźną strukturą liniową (bruzdy ust).
"""

from __future__ import annotations

import numpy as np
from skimage.feature import hog


def extract_hog(
    gray_image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
) -> tuple[np.ndarray, np.ndarray]:
    """Oblicz cechy HOG z obrazu w skali szarości.

    Args:
        gray_image: obraz grayscale (2D uint8). Najlepiej po resize do 512×256.
        orientations: liczba kierunków gradientu (bins kątowych).
        pixels_per_cell: rozmiar komórki HOG w pikselach (w, h).
        cells_per_block: liczba komórek w bloku (dla normalizacji).

    Returns:
        Krotka (hog_image, feature_vector):
            hog_image — wizualizacja gradientów (do wyświetlenia),
            feature_vector — wektor cech HOG (float32).
    """
    feature_vector, hog_image = hog(
        gray_image,
        orientations=orientations,
        pixels_per_cell=pixels_per_cell,
        cells_per_block=cells_per_block,
        visualize=True,
        feature_vector=True,
    )
    return hog_image.astype(np.float32), feature_vector.astype(np.float32)


def hog_feature_vector(
    gray_image: np.ndarray,
    orientations: int = 9,
    pixels_per_cell: tuple[int, int] = (16, 16),
    cells_per_block: tuple[int, int] = (2, 2),
) -> np.ndarray:
    """Zwróć tylko wektor cech HOG (bez wizualizacji)."""
    _, fv = extract_hog(gray_image, orientations, pixels_per_cell, cells_per_block)
    return fv
