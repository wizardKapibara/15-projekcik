"""Centralne ścieżki i stałe konfiguracyjne projektu."""

from __future__ import annotations

from pathlib import Path

# Korzeń projektu (folder zawierający 'src', 'data', 'apps', ...)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dane wejściowe
DATA_DIR = PROJECT_ROOT / "data"
BAZA_DANYCH_DIR = DATA_DIR / "baza_danych"
METADATA_XLSM = DATA_DIR / "10 osoby - ok.xlsm"

# Dane pośrednie / wyjściowe
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
EXPORTS_DIR = PROJECT_ROOT / "exports"

# Reprodukowalność
RANDOM_SEED = 42

# Liczba foldów cross-validation
N_SPLITS = 5

# Docelowy rozmiar obrazu po preprocessingu
TARGET_SIZE = (512, 256)  # (width, height)


def ensure_dirs() -> None:
    """Tworzy wszystkie potrzebne foldery wyjściowe jeśli nie istnieją."""
    for d in (PROCESSED_DIR, SPLITS_DIR, MODELS_DIR, RESULTS_DIR, PLOTS_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
