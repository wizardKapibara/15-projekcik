"""Ekstrakcja cech przez bank filtrów Gabora.

Filtry Gabora są optymalnym narzędziem do analizy tekstur kierunkowych
(np. bruzdy ust w różnych orientacjach). Bank filtrów łączy kilka orientacji
i częstotliwości, dając bogatą charakterystykę tekstury.

Dla każdego filtra obliczamy: średnia odpowiedzi + odchylenie standardowe.
Bank 8 orientacji × 4 częstotliwości = 32 filtry → wektor długości 64.
"""

from __future__ import annotations

import cv2
import numpy as np


def build_gabor_bank(
    orientations: int = 8,
    frequencies: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    kernel_size: int = 21,
    sigma: float = 4.0,
    gamma: float = 0.5,
) -> list[np.ndarray]:
    """Zbuduj bank jąder filtrów Gabora.

    Args:
        orientations: liczba kierunków (0° do 180°).
        frequencies: częstotliwości fali nośnej (cycles/pixel).
        kernel_size: rozmiar jądra (nieparzyste).
        sigma: sigma gaussowskiej koperty (szerokość odpowiedzi).
        gamma: współczynnik proporcji elipsy jądra.

    Returns:
        Lista jąder (numpy arrays) gotowych do cv2.filter2D.
    """
    kernels: list[np.ndarray] = []
    for freq in frequencies:
        lam = 1.0 / freq  # długość fali
        for i in range(orientations):
            theta = np.pi * i / orientations
            kernel = cv2.getGaborKernel(
                ksize=(kernel_size, kernel_size),
                sigma=sigma,
                theta=theta,
                lambd=lam,
                gamma=gamma,
                psi=0,
                ktype=cv2.CV_32F,
            )
            kernel /= kernel.sum() if kernel.sum() != 0 else 1.0  # normalizacja
            kernels.append(kernel)
    return kernels


# Globalny bank filtrów — tworzony raz (drogie obliczeniowo)
_DEFAULT_BANK: list[np.ndarray] | None = None


def _get_default_bank() -> list[np.ndarray]:
    global _DEFAULT_BANK
    if _DEFAULT_BANK is None:
        _DEFAULT_BANK = build_gabor_bank()
    return _DEFAULT_BANK


def extract_gabor(
    gray_image: np.ndarray,
    orientations: int = 8,
    frequencies: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    kernel_size: int = 21,
    sigma: float = 4.0,
    gamma: float = 0.5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Zastosuj bank filtrów Gabora i oblicz wektor cech.

    Args:
        gray_image: obraz grayscale (uint8 lub float32).
        orientations: liczba kierunków w banku.
        frequencies: krotka częstotliwości.
        kernel_size: rozmiar jądra.
        sigma: sigma gaussowskiej koperty.
        gamma: proporcja elipsy.

    Returns:
        Krotka (responses, feature_vector):
            responses — lista obrazów odpowiedzi (po jednym na filtr),
            feature_vector — wektor [mean_0, std_0, mean_1, std_1, ...] float32.
    """
    img_f = gray_image.astype(np.float32)
    kernels = build_gabor_bank(orientations, frequencies, kernel_size, sigma, gamma)

    responses: list[np.ndarray] = []
    features: list[float] = []

    for kernel in kernels:
        response = cv2.filter2D(img_f, cv2.CV_32F, kernel)
        responses.append(np.abs(response))  # energia odpowiedzi
        features.append(float(np.mean(np.abs(response))))
        features.append(float(np.std(np.abs(response))))

    return responses, np.array(features, dtype=np.float32)


def gabor_feature_vector(
    gray_image: np.ndarray,
    orientations: int = 8,
    frequencies: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4),
    **kwargs,
) -> np.ndarray:
    """Zwróć tylko wektor cech Gabora (bez listy odpowiedzi)."""
    _, fv = extract_gabor(gray_image, orientations=orientations, frequencies=frequencies, **kwargs)
    return fv


def gabor_responses_grid(responses: list[np.ndarray], orientations: int = 8) -> np.ndarray:
    """Złóż odpowiedzi filtrów w siatkę obrazów (do wizualizacji).

    Returns:
        Jeden obraz ze wszystkimi odpowiedziami ułożonymi w siatce
        (wiersze = częstotliwości, kolumny = orientacje).
    """
    n_filters = len(responses)
    n_freq = n_filters // orientations
    rows: list[np.ndarray] = []
    for fi in range(n_freq):
        row_imgs: list[np.ndarray] = []
        for oi in range(orientations):
            img = responses[fi * orientations + oi]
            # Normalizuj do 0-255
            mn, mx = img.min(), img.max()
            if mx > mn:
                norm = ((img - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                norm = np.zeros_like(img, dtype=np.uint8)
            row_imgs.append(norm)
        rows.append(np.hstack(row_imgs))
    return np.vstack(rows)
