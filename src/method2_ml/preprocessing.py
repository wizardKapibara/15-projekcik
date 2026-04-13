"""Preprocessing obrazów ust — pipeline krok po kroku.

Pipeline:
    1. Oryginał (wczytany)
    2. Skala szarości
    3. CLAHE (adaptacyjne wyrównanie histogramu)
    4. Bilateral filter (redukcja szumu z zachowaniem krawędzi)
    5. Resize do docelowego rozmiaru (domyślnie 512×256)

Każdy krok zwracany osobno jako numpy array, żeby można go wyświetlić w GUI.
Opcjonalny zapis pośrednich wyników do folderu cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from src.core.config import PROCESSED_DIR, TARGET_SIZE
from src.core.dataset_loader import LipDataset
from src.core.filename_parser import try_parse_filename


@dataclass
class PreprocessingResult:
    """Wyniki każdego kroku pipeline preprocessingu."""

    original_bgr: np.ndarray      # oryginał w BGR (do wyświetlenia w Streamlit jako RGB)
    grayscale: np.ndarray         # skala szarości
    clahe: np.ndarray             # po CLAHE
    denoised: np.ndarray          # po bilateral filter
    resized: np.ndarray           # po resize (końcowy obraz do ekstrakcji cech)

    # Użyte parametry (do wyświetlenia w GUI)
    clahe_clip_limit: float
    clahe_tile_size: int
    bilateral_d: int
    bilateral_sigma_color: float
    bilateral_sigma_space: float
    target_size: Tuple[int, int]

    @property
    def original_rgb(self) -> np.ndarray:
        """Oryginał w RGB (gotowy do st.image)."""
        return cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)


def run_pipeline(
    image_bgr: np.ndarray,
    clahe_clip_limit: float = 2.0,
    clahe_tile_size: int = 8,
    bilateral_d: int = 9,
    bilateral_sigma_color: float = 75.0,
    bilateral_sigma_space: float = 75.0,
    target_size: Tuple[int, int] = TARGET_SIZE,
) -> PreprocessingResult:
    """Wykonaj pełny pipeline preprocessingu na obrazie BGR.

    Args:
        image_bgr: obraz wejściowy w formacie BGR (z cv2).
        clahe_clip_limit: limit kontrastu dla CLAHE (im wyższy, tym mocniejszy kontrast).
        clahe_tile_size: rozmiar kafla dla CLAHE.
        bilateral_d: średnica otoczenia dla bilateral filter.
        bilateral_sigma_color: sigma filtra w przestrzeni koloru.
        bilateral_sigma_space: sigma filtra w przestrzeni geometrycznej.
        target_size: docelowy rozmiar (width, height).

    Returns:
        PreprocessingResult z obrazami po każdym kroku.
    """
    # Krok 2: skala szarości
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Krok 3: CLAHE
    clahe_obj = cv2.createCLAHE(
        clipLimit=clahe_clip_limit,
        tileGridSize=(clahe_tile_size, clahe_tile_size),
    )
    clahe_img = clahe_obj.apply(gray)

    # Krok 4: bilateral filter (redukcja szumu z zachowaniem krawędzi)
    denoised = cv2.bilateralFilter(
        clahe_img,
        d=bilateral_d,
        sigmaColor=bilateral_sigma_color,
        sigmaSpace=bilateral_sigma_space,
    )

    # Krok 5: resize do docelowego rozmiaru
    resized = cv2.resize(denoised, target_size, interpolation=cv2.INTER_AREA)

    return PreprocessingResult(
        original_bgr=image_bgr,
        grayscale=gray,
        clahe=clahe_img,
        denoised=denoised,
        resized=resized,
        clahe_clip_limit=clahe_clip_limit,
        clahe_tile_size=clahe_tile_size,
        bilateral_d=bilateral_d,
        bilateral_sigma_color=bilateral_sigma_color,
        bilateral_sigma_space=bilateral_sigma_space,
        target_size=target_size,
    )


def run_pipeline_from_path(
    path: Path | str,
    **kwargs,
) -> PreprocessingResult:
    """Wykonaj pipeline z pliku na dysku.

    Wygodna nakładka na run_pipeline z ładowaniem pliku.
    """
    path = Path(path)
    image_bgr = LipDataset.load_image(path, color="bgr")
    return run_pipeline(image_bgr, **kwargs)


def save_steps_to_cache(
    result: PreprocessingResult,
    image_path: Path | str,
    cache_dir: Path | str = PROCESSED_DIR,
) -> dict[str, Path]:
    """Zapisz obrazy pośrednie do folderu cache.

    Tworzy strukturę: <cache_dir>/<person>/<image_stem>/01_original.png ...

    Args:
        result: wynik pipeline.
        image_path: oryginalna ścieżka pliku (do wyznaczenia ścieżki cache).
        cache_dir: folder bazowy cache.

    Returns:
        Słownik krok → ścieżka pliku.
    """
    image_path = Path(image_path)
    cache_dir = Path(cache_dir)

    parsed = try_parse_filename(image_path)
    if parsed:
        person_folder = f"{parsed.person_id:02d}"
    else:
        person_folder = "unknown"

    stem = image_path.stem
    out_dir = cache_dir / person_folder / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}

    steps = {
        "01_original": cv2.cvtColor(result.original_bgr, cv2.COLOR_BGR2RGB),
        "02_grayscale": result.grayscale,
        "03_clahe": result.clahe,
        "04_denoised": result.denoised,
        "05_resized": result.resized,
    }

    for name, img in steps.items():
        out_path = out_dir / f"{name}.png"
        if img.ndim == 2:
            # Grayscale — zapisz bezpośrednio
            cv2.imwrite(str(out_path), img)
        else:
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved[name] = out_path

    return saved


def export_steps_to_documentation(
    result: PreprocessingResult,
    image_path: Path | str,
    export_dir: Path | str,
    prefix: str = "",
) -> list[Path]:
    """Eksportuj obrazy pośrednie do folderu dokumentacji (czytelne nazwy).

    Tworzy pliki: <prefix>_01_oryginał.png, <prefix>_02_skala_szarosci.png, ...

    Args:
        result: wynik pipeline.
        image_path: oryginalna ścieżka pliku (dla prefiksu jeśli nie podany).
        export_dir: folder docelowy (zostanie utworzony).
        prefix: opcjonalny prefiks nazwy pliku.

    Returns:
        Lista ścieżek wyeksportowanych plików.
    """
    image_path = Path(image_path)
    export_dir = Path(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    if not prefix:
        prefix = image_path.stem

    exported: list[Path] = []

    steps = [
        ("01_oryginal", result.original_rgb),
        ("02_skala_szarosci", result.grayscale),
        ("03_clahe", result.clahe),
        ("04_bilateral_filter", result.denoised),
        ("05_resize", result.resized),
    ]

    for label, img in steps:
        out_path = export_dir / f"{prefix}_{label}.png"
        if img.ndim == 2:
            cv2.imwrite(str(out_path), img)
        else:
            cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        exported.append(out_path)

    return exported
