"""Ekstrakcja minucji bruzd ust (cheiloskopia).

Analogia do daktyloskopii:
    - Binaryzacja: oddzielenie bruzd od tła
    - Skeletonizacja: redukcja bruzd do linii o szerokości 1 piksela
    - Detekcja minucji metodą Crossing Number:
        * Ending (zakończenie) — piksel z 1 sąsiadem w szkielecie
        * Bifurcation (rozwidlenie) — piksel z 3 sąsiadami
        * Crossing (skrzyżowanie) — piksel z 4 sąsiadami
    - Statystyki globalne → wektor cech

Klasyfikacja Suzuki i Tsuchihashi (1970): typy bruzd ust są analogiczne
do typów linii papilarnych i mogą identyfikować osobę.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np
from skimage.morphology import skeletonize


# Typy minucji
MinutiaeType = Literal["ending", "bifurcation", "crossing", "other"]


@dataclass
class Minutia:
    """Pojedyncza minucja (punkt charakterystyczny bruzdy)."""

    x: int
    y: int
    minutiae_type: MinutiaeType
    crossing_number: int  # surowa wartość CN


@dataclass
class MinutiaeResult:
    """Wyniki ekstrakcji minucji dla jednego obrazu."""

    binary: np.ndarray          # obraz binarny (bruzdy = 255, tło = 0)
    skeleton: np.ndarray        # szkielet (cienkie bruzdy = 255)
    minutiae: list[Minutia]     # wykryte minucje

    # Statystyki globalne (wektor cech)
    n_endings: int
    n_bifurcations: int
    n_crossings: int
    n_total_minutiae: int
    groove_density: float        # minucje / 10 000 pikseli szkieletu
    total_skeleton_pixels: int   # całkowita długość bruzd (w pikselach)
    binary_entropy: float        # entropia Shannona obrazu binarnego

    def feature_vector(self) -> np.ndarray:
        """Zwróć wektor cech jako float32 array."""
        return np.array([
            self.n_endings,
            self.n_bifurcations,
            self.n_crossings,
            self.n_total_minutiae,
            self.groove_density,
            self.total_skeleton_pixels,
            self.binary_entropy,
        ], dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        return [
            "n_endings", "n_bifurcations", "n_crossings", "n_total_minutiae",
            "groove_density", "total_skeleton_pixels", "binary_entropy",
        ]


def _crossing_number(neighborhood: np.ndarray) -> int:
    """Oblicz Crossing Number dla piksela szkieletu.

    CN = 0.5 * sum(|p_i - p_{i+1}|) dla i = 1..8 (cyklicznie).
    Standardowe znaczenie:
        CN=1 → ending, CN=2 → continuing, CN=3 → bifurcation, CN=4 → crossing.
    """
    p = neighborhood.ravel()
    # Kolejność sąsiadów 3×3 zgodnie z ruchem wskazówek zegara:
    # p2 p3 p4
    # p1  c p5
    # p8 p7 p6
    order = [p[1], p[2], p[5], p[8], p[7], p[6], p[3], p[0]]
    cn = sum(abs(int(order[i]) - int(order[(i + 1) % 8])) for i in range(8))
    return cn // 2


def binarize(
    gray_image: np.ndarray,
    method: Literal["adaptive", "otsu", "gabor_otsu"] = "adaptive",
    block_size: int = 35,
    c_constant: int = 10,
) -> np.ndarray:
    """Binaryzacja obrazu w skali szarości.

    Args:
        gray_image: obraz grayscale (uint8).
        method:
            'adaptive' — adaptacyjna binaryzacja (zalecana dla zmiennego oświetlenia),
            'otsu' — próg Otsu (globalny, szybszy),
            'gabor_otsu' — Otsu + enhancement Gaborem (najlepsza jakość, wolniejszy).
        block_size: rozmiar bloku dla adaptive threshold (musi być nieparzysty).
        c_constant: stała odejmowana od średniej lokalnej (adaptive).

    Returns:
        Obraz binarny: bruzdy = 255, tło = 0.
    """
    if method == "adaptive":
        binary = cv2.adaptiveThreshold(
            gray_image,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=block_size if block_size % 2 == 1 else block_size + 1,
            C=c_constant,
        )
    elif method == "otsu":
        _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    elif method == "gabor_otsu":
        # Wzmocnienie kontrastu bruzd filtrem Gabora przed binaryzacją
        kernel = cv2.getGaborKernel((21, 21), 4.0, np.pi / 4, 8.0, 0.5, 0, ktype=cv2.CV_32F)
        enhanced = cv2.filter2D(gray_image.astype(np.float32), cv2.CV_32F, kernel)
        enhanced = np.abs(enhanced)
        enhanced = ((enhanced / enhanced.max()) * 255).astype(np.uint8) if enhanced.max() > 0 else gray_image
        _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        raise ValueError(f"Nieznana metoda binaryzacji: {method!r}")

    # Morfologiczne czyszczenie — usunięcie małych artefaktów
    kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_morph)
    return binary


def _binary_entropy(binary: np.ndarray) -> float:
    """Entropia Shannona obrazu binarnego (mierzy złożoność wzorca bruzd)."""
    total = binary.size
    n_white = np.count_nonzero(binary)
    n_black = total - n_white
    if n_white == 0 or n_black == 0:
        return 0.0
    p_w = n_white / total
    p_b = n_black / total
    return float(-p_w * np.log2(p_w) - p_b * np.log2(p_b))


def extract_minutiae(
    gray_image: np.ndarray,
    binarize_method: Literal["adaptive", "otsu", "gabor_otsu"] = "adaptive",
    block_size: int = 35,
    c_constant: int = 10,
    border_margin: int = 10,
) -> MinutiaeResult:
    """Pełny pipeline ekstrakcji minucji: binaryzacja → szkielet → minucje.

    Args:
        gray_image: obraz grayscale (uint8), najlepiej po preprocessingu.
        binarize_method: metoda binaryzacji.
        block_size: parametr adaptive threshold.
        c_constant: parametr adaptive threshold.
        border_margin: margines brzegowy (px) do wykluczenia artefaktów.

    Returns:
        MinutiaeResult z obrazami pośrednimi i wektorem cech.
    """
    # Krok 1: binaryzacja
    binary = binarize(gray_image, method=binarize_method,
                      block_size=block_size, c_constant=c_constant)

    # Krok 2: skeletonizacja (redukcja bruzd do linii 1-pikselowej)
    binary_bool = binary.astype(bool)
    skeleton_bool = skeletonize(binary_bool)
    skeleton = (skeleton_bool * 255).astype(np.uint8)

    # Krok 3: wykrywanie minucji metodą Crossing Number
    h, w = skeleton.shape
    minutiae: list[Minutia] = []
    skel_bin = skeleton_bool.astype(np.uint8)

    for y in range(border_margin, h - border_margin):
        for x in range(border_margin, w - border_margin):
            if skel_bin[y, x] == 0:
                continue
            neighborhood = skel_bin[y - 1:y + 2, x - 1:x + 2].copy()
            neighborhood[1, 1] = 0  # Wyzeruj środkowy piksel przy liczeniu
            cn = _crossing_number(neighborhood)
            if cn == 1:
                mtype: MinutiaeType = "ending"
            elif cn == 3:
                mtype = "bifurcation"
            elif cn >= 4:
                mtype = "crossing"
            else:
                continue  # CN=2: zwykły punkt linii — pomijamy
            minutiae.append(Minutia(x=x, y=y, minutiae_type=mtype, crossing_number=cn))

    # Statystyki
    n_endings = sum(1 for m in minutiae if m.minutiae_type == "ending")
    n_bifurcations = sum(1 for m in minutiae if m.minutiae_type == "bifurcation")
    n_crossings = sum(1 for m in minutiae if m.minutiae_type == "crossing")
    total_skeleton = int(skel_bin.sum())
    density = (len(minutiae) / total_skeleton * 10_000) if total_skeleton > 0 else 0.0

    return MinutiaeResult(
        binary=binary,
        skeleton=skeleton,
        minutiae=minutiae,
        n_endings=n_endings,
        n_bifurcations=n_bifurcations,
        n_crossings=n_crossings,
        n_total_minutiae=len(minutiae),
        groove_density=float(density),
        total_skeleton_pixels=total_skeleton,
        binary_entropy=_binary_entropy(binary),
    )


def draw_minutiae(
    skeleton: np.ndarray,
    minutiae: list[Minutia],
    radius: int = 4,
) -> np.ndarray:
    """Narysuj minucje jako kolorowe kółka na szkielecie.

    Kolory:
        Zielony (#00FF00) — ending (zakończenie)
        Czerwony (#FF0000) — bifurcation (rozwidlenie)
        Niebieski (#0080FF) — crossing (skrzyżowanie)

    Args:
        skeleton: obraz szkieletu (grayscale lub RGB).
        minutiae: lista minucji do narysowania.
        radius: promień kółka markera.

    Returns:
        Obraz RGB z narysowanymi minucjami.
    """
    if skeleton.ndim == 2:
        vis = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    else:
        vis = skeleton.copy()

    colors = {
        "ending": (0, 200, 0),       # zielony
        "bifurcation": (220, 30, 30), # czerwony
        "crossing": (30, 100, 220),   # niebieski
    }

    for m in minutiae:
        color = colors.get(m.minutiae_type, (180, 180, 0))
        cv2.circle(vis, (m.x, m.y), radius, color, -1)

    return vis
