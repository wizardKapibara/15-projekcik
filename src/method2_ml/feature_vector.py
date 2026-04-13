"""Budowanie i zarządzanie kompletnym wektorem cech.

Wektor = konkatenacja: LBP (10) + HOG (→PCA→100) + Gabor (64) + Minucje (7)
Razem: ~181 cech po redukcji PCA dla HOG.

PCA dla HOG jest uzasadnione naukowo:
- Surowy HOG: 16740 cech przy 382 próbkach → curse of dimensionality
- PCA redukuje do 100 komponentów zachowując >95% wariancji
- Klasyfikator uczy się szybciej i generaluje lepiej

Zarządzanie cache: wyekstrahowane wektory zapisywane do .npy żeby nie
przeliczać za każdym razem (Gabor dla 382 zdjęć zajmuje ok. 30 sekund).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.core.config import PROCESSED_DIR, RANDOM_SEED
from src.core.filename_parser import try_parse_filename
from src.method2_ml.features_gabor import gabor_feature_vector
from src.method2_ml.features_hog import hog_feature_vector
from src.method2_ml.features_lbp import lbp_feature_vector
from src.method2_ml.features_minutiae import extract_minutiae
from src.method2_ml.preprocessing import run_pipeline


HOG_PCA_COMPONENTS = 100  # Liczba składowych PCA dla HOG


def extract_raw_features(gray_resized: np.ndarray) -> dict[str, np.ndarray]:
    """Wyekstrahuj surowe cechy (przed skalowaniem/PCA).

    Args:
        gray_resized: obraz grayscale po preprocessingu (256×512).

    Returns:
        Słownik {nazwa: wektor} dla każdego ekstraktora.
    """
    return {
        "lbp": lbp_feature_vector(gray_resized),
        "hog": hog_feature_vector(gray_resized),
        "gabor": gabor_feature_vector(gray_resized),
        "minutiae": extract_minutiae(gray_resized).feature_vector(),
    }


# Cache wymiarów — policzone raz, przy pierwszym wywołaniu
_FEATURE_DIMS_CACHE: Optional[dict[str, int]] = None


def get_feature_dims() -> dict[str, int]:
    """Zwróć wymiary poszczególnych ekstraktorów (lazy, cached).

    Zamiast re-ładować pierwsze zdjęcie treningowe w fit(), liczymy wymiary
    raz na dummy image i cache'ujemy.
    """
    global _FEATURE_DIMS_CACHE
    if _FEATURE_DIMS_CACHE is None:
        from src.core.config import TARGET_SIZE
        dummy = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
        raw = extract_raw_features(dummy)
        _FEATURE_DIMS_CACHE = {k: len(v) for k, v in raw.items()}
    return _FEATURE_DIMS_CACHE


def _get_cache_path(image_path: Path | str) -> Path:
    """Ścieżka do pliku cache wektora cech (.npy)."""
    image_path = Path(image_path)
    parsed = try_parse_filename(image_path)
    person_folder = f"{parsed.person_id:02d}" if parsed else "unknown"
    stem = image_path.stem
    cache_dir = PROCESSED_DIR / person_folder / stem
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "features.npy"


def extract_features_with_cache(
    image_path: Path | str,
    force_recompute: bool = False,
) -> np.ndarray:
    """Wyekstrahuj surowe cechy z cache lub oblicz na nowo.

    Cache: data/processed/<person>/<stem>/features.npy
    Wektor zawiera SUROWE cechy (bez PCA/skalowania) do późniejszego FIT scaler/PCA.

    Args:
        image_path: ścieżka do pliku PNG.
        force_recompute: ignoruj cache i oblicz od nowa.

    Returns:
        Surowy wektor cech (float32), np. długości 16821.
    """
    image_path = Path(image_path)
    cache_path = _get_cache_path(image_path)

    # Sprawdź cache (tylko jeśli plik cache jest nowszy niż obraz źródłowy)
    if not force_recompute and cache_path.exists():
        if cache_path.stat().st_mtime >= image_path.stat().st_mtime:
            return np.load(cache_path)

    # Oblicz od nowa
    from src.core.dataset_loader import LipDataset
    image_bgr = LipDataset.load_image(image_path, color="bgr")
    raw = extract_raw_features(run_pipeline(image_bgr).resized)

    # Konkatenacja w ustalonym porządku
    fv = np.concatenate([raw["lbp"], raw["hog"], raw["gabor"], raw["minutiae"]]).astype(np.float32)
    np.save(cache_path, fv)
    return fv


class FeatureExtractorPipeline:
    """Pipeline: surowe cechy → PCA (HOG) → StandardScaler → gotowy wektor.

    Użycie:
        pipeline = FeatureExtractorPipeline(hog_pca_components=100)
        pipeline.fit(train_paths)
        X_train = pipeline.transform(train_paths)
        X_test = pipeline.transform(test_paths)
    """

    def __init__(self, hog_pca_components: int = HOG_PCA_COMPONENTS) -> None:
        self.hog_pca_components = hog_pca_components
        self._scaler = StandardScaler()
        self._pca: Optional[PCA] = None
        self._fitted = False

        # Wymiary — raz policzone przy fit() z pomocą get_feature_dims()
        self._lbp_dim: int = 0
        self._hog_dim: int = 0
        self._gabor_dim: int = 0
        self._minutiae_dim: int = 0

    @property
    def feature_names(self) -> list[str]:
        names = [f"lbp_{i}" for i in range(self._lbp_dim)]
        if self.hog_pca_components > 0:
            names += [f"hog_pc{i}" for i in range(self.hog_pca_components)]
        else:
            names += [f"hog_{i}" for i in range(self._hog_dim)]
        names += [f"gabor_{i}" for i in range(self._gabor_dim)]
        names += ["n_endings", "n_bifurcations", "n_crossings", "n_total_minutiae",
                  "groove_density", "total_skeleton_pixels", "binary_entropy"]
        return names

    def _apply_pca_and_concat(self, X_raw: np.ndarray) -> np.ndarray:
        """Zastosuj PCA na części HOG i skleuj wektor."""
        n = X_raw.shape[0]
        parts: List[np.ndarray] = []

        lbp = X_raw[:, :self._lbp_dim]
        hog = X_raw[:, self._lbp_dim:self._lbp_dim + self._hog_dim]
        gabor = X_raw[:, self._lbp_dim + self._hog_dim:self._lbp_dim + self._hog_dim + self._gabor_dim]
        minutiae = X_raw[:, self._lbp_dim + self._hog_dim + self._gabor_dim:]

        parts.append(lbp)
        if self.hog_pca_components > 0 and self._pca is not None:
            parts.append(self._pca.transform(hog))
        else:
            parts.append(hog)
        parts.append(gabor)
        parts.append(minutiae)

        return np.hstack(parts)

    def fit(
        self,
        train_paths: List[str | Path],
        force_recompute: bool = False,
        progress_callback=None,
    ) -> "FeatureExtractorPipeline":
        """Fit scaler i PCA na danych treningowych.

        Args:
            train_paths: lista ścieżek do zdjęć treningowych.
            force_recompute: wymuś ponowną ekstrakcję (ignoruj cache).
            progress_callback: opcjonalnie funkcja(i, n) wywoływana co obraz (dla GUI).
        """
        # Ustaw wymiary BEZ ponownego ładowania zdjęcia (cached global)
        dims = get_feature_dims()
        self._lbp_dim = dims["lbp"]
        self._hog_dim = dims["hog"]
        self._gabor_dim = dims["gabor"]
        self._minutiae_dim = dims["minutiae"]

        # Załaduj surowe wektory (z cache .npy jeśli są)
        raw_list: List[np.ndarray] = []
        for i, p in enumerate(train_paths):
            raw_list.append(extract_features_with_cache(p, force_recompute=force_recompute))
            if progress_callback:
                progress_callback(i + 1, len(train_paths))

        X_raw = np.array(raw_list, dtype=np.float32)

        # PCA dla HOG
        if self.hog_pca_components > 0:
            hog_part = X_raw[:, self._lbp_dim:self._lbp_dim + self._hog_dim]
            n_components = min(self.hog_pca_components, hog_part.shape[0], hog_part.shape[1])
            self._pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
            self._pca.fit(hog_part)

        # Buduj macierz po PCA, potem fit scaler
        X_reduced = self._apply_pca_and_concat(X_raw)
        self._scaler.fit(X_reduced)
        self._fitted = True
        return self

    def transform(
        self,
        paths: List[str | Path],
        force_recompute: bool = False,
        progress_callback=None,
    ) -> np.ndarray:
        """Transformuj listę ścieżek → gotowa macierz cech (skalowana).

        Args:
            paths: lista ścieżek.
            force_recompute: wymuś ponowną ekstrakcję.
            progress_callback: opcjonalnie funkcja(i, n).

        Returns:
            X (n_samples × n_features), float32, zestandaryzowany.
        """
        if not self._fitted:
            raise RuntimeError("Najpierw wywołaj fit().")

        raw_list: List[np.ndarray] = []
        for i, p in enumerate(paths):
            raw_list.append(extract_features_with_cache(p, force_recompute=force_recompute))
            if progress_callback:
                progress_callback(i + 1, len(paths))

        X_raw = np.array(raw_list, dtype=np.float32)
        X_reduced = self._apply_pca_and_concat(X_raw)
        return self._scaler.transform(X_reduced).astype(np.float32)

    def transform_single(self, image_path: str | Path, force_recompute: bool = False) -> np.ndarray:
        """Transformuj jedno zdjęcie → wektor cech (1D)."""
        return self.transform([image_path], force_recompute=force_recompute)[0]

    def save(self, path: Path | str = None) -> Path:
        """Zapisz pipeline (scaler + PCA) do pliku joblib."""
        import joblib as jl
        from src.core.config import MODELS_DIR
        if path is None:
            path = MODELS_DIR / "method2_pipeline.joblib"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        jl.dump(self, str(path))
        return Path(path)

    @classmethod
    def load(cls, path: Path | str = None) -> "FeatureExtractorPipeline":
        """Wczytaj pipeline z pliku joblib."""
        import joblib as jl
        from src.core.config import MODELS_DIR
        if path is None:
            path = MODELS_DIR / "method2_pipeline.joblib"
        if not Path(path).exists():
            raise FileNotFoundError(f"Brak pliku pipeline: {path}")
        return jl.load(str(path))
