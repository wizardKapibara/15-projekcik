"""Klasyfikatory SVM / Random Forest / k-NN dla biometrii ust.

Trenowanie, predykcja (top-1, top-5), zapis/odczyt modeli.
Wrappery są symetryczne: jednolite API niezależnie od wyboru klasyfikatora.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from src.core.config import MODELS_DIR, RANDOM_SEED

AVAILABLE_CLASSIFIERS = ["svm", "rf", "knn"]

_MODEL_FILENAMES = {
    "svm": "method2_svm.joblib",
    "rf": "method2_rf.joblib",
    "knn": "method2_knn.joblib",
}


def _build_classifier(name: str):
    """Zbuduj nowy obiekt klasyfikatora z domyślnymi parametrami."""
    if name == "svm":
        return SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            random_state=RANDOM_SEED,
            class_weight="balanced",
        )
    elif name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            max_features="sqrt",
            random_state=RANDOM_SEED,
            class_weight="balanced",
            n_jobs=-1,
        )
    elif name == "knn":
        return KNeighborsClassifier(
            n_neighbors=15,
            weights="uniform",
            metric="euclidean",
            n_jobs=-1,
        )
    else:
        raise ValueError(f"Nieznany klasyfikator: {name!r}. Dostępne: {AVAILABLE_CLASSIFIERS}")


class LipClassifier:
    """Wrapper dla klasyfikatora identyfikacji osoby po bruzdach ust.

    Obsługuje trzy modele: SVM (rbf), Random Forest, k-NN.
    Wszystkie zwracają top-1, top-5 i confidence per klasa.
    """

    def __init__(self, model_name: str = "svm") -> None:
        """
        Args:
            model_name: 'svm', 'rf', lub 'knn'.
        """
        if model_name not in AVAILABLE_CLASSIFIERS:
            raise ValueError(f"Nieznany model: {model_name!r}. Dostępne: {AVAILABLE_CLASSIFIERS}")
        self.model_name = model_name
        self._clf = _build_classifier(model_name)
        self._fitted = False
        self._classes_: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def classes(self) -> Optional[np.ndarray]:
        return self._classes_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LipClassifier":
        """Trenuj klasyfikator.

        Args:
            X: macierz cech (n_samples × n_features), zestandaryzowana.
            y: etykiety (person_id 1-22).
        """
        self._clf.fit(X, y)
        self._classes_ = np.unique(y)
        self._fitted = True
        return self

    def predict_top1(self, X: np.ndarray) -> np.ndarray:
        """Predykcja top-1 dla każdej próbki."""
        if not self._fitted:
            raise RuntimeError("Model nie jest wytrenowany.")
        if self.model_name == "knn":
            # Spójność z _knn_softmax_proba: argmax z naszych proba
            proba = self.predict_proba(X)
            return self._clf.classes_[np.argmax(proba, axis=1)]
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Prawdopodobieństwa per klasa (n_samples × n_classes).

        Dla k-NN używamy softmax z odległości do najbliższej próbki każdej klasy
        zamiast głosowania sąsiadów — w biometrii cechy jednej osoby są ciasno
        zgrupowane, więc zwykłe `predict_proba` k-NN dawałoby 100% dla dominującej
        klasy (mało informatywne).
        """
        if not self._fitted:
            raise RuntimeError("Model nie jest wytrenowany.")
        if self.model_name == "knn":
            return self._knn_softmax_proba(X)
        return self._clf.predict_proba(X)

    def _knn_softmax_proba(self, X: np.ndarray) -> np.ndarray:
        """Softmax z odległości min-per-class.

        Dla każdej próbki zapytania liczymy odległość euklidesową do wszystkich
        próbek treningowych, bierzemy minimum per klasa, i robimy softmax(-d/T),
        gdzie T to mediana tych minimów (autoskalowanie temperatury).
        """
        train_X = self._clf._fit_X
        train_y = self._clf._y
        classes = self._clf.classes_
        n_classes = len(classes)

        D = pairwise_distances(X, train_X, metric="euclidean")
        n_queries = X.shape[0]
        min_d = np.full((n_queries, n_classes), np.inf, dtype=np.float64)
        for c_idx in range(n_classes):
            mask = (train_y == c_idx)
            if mask.any():
                min_d[:, c_idx] = D[:, mask].min(axis=1)

        # Temperatura: mediana minimalnych odległości per query (autoskalowanie)
        temp = np.median(min_d, axis=1, keepdims=True)
        temp = np.clip(temp, 1e-6, None)
        logits = -min_d / temp
        logits -= logits.max(axis=1, keepdims=True)  # stabilność numeryczna
        exp = np.exp(logits)
        proba = exp / exp.sum(axis=1, keepdims=True)
        return proba

    def predict_single(
        self, x: np.ndarray, top_k: int = 5
    ) -> Dict:
        """Predykcja dla jednej próbki z top-k kandydatami.

        Args:
            x: wektor cech (1D).
            top_k: liczba zwracanych kandydatów.

        Returns:
            Słownik:
                'predicted': int — top-1 (person_id),
                'top_k': List[Tuple[int, float]] — [(person_id, confidence), ...],
                'confidence': float — pewność top-1,
                'all_proba': dict {person_id: confidence}.
        """
        if not self._fitted:
            raise RuntimeError("Model nie jest wytrenowany.")

        x2d = x.reshape(1, -1)
        proba = self.predict_proba(x2d)[0]  # (n_classes,) — użyj własnego wrappera (k-NN softmax)
        classes = self._clf.classes_

        # Sortuj po malejącym confidence
        sorted_idx = np.argsort(proba)[::-1]
        top_k_list = [(int(classes[i]), float(proba[i])) for i in sorted_idx[:top_k]]

        return {
            "predicted": top_k_list[0][0],
            "confidence": top_k_list[0][1],
            "top_k": top_k_list,
            "all_proba": {int(classes[i]): float(proba[i]) for i in range(len(classes))},
        }

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Accuracy na zbiorze (X, y)."""
        return float(self._clf.score(X, y))

    def top_k_accuracy(self, X: np.ndarray, y: np.ndarray, k: int = 5) -> float:
        """Top-k accuracy na zbiorze."""
        proba = self.predict_proba(X)
        classes = self._clf.classes_
        correct = 0
        for i, true_label in enumerate(y):
            top_k_idx = np.argsort(proba[i])[::-1][:k]
            if true_label in classes[top_k_idx]:
                correct += 1
        return correct / len(y)

    def save(self, models_dir: Path = MODELS_DIR) -> Path:
        """Zapisz wytrenowany model do pliku joblib."""
        if not self._fitted:
            raise RuntimeError("Model nie jest wytrenowany — nie można zapisać.")
        models_dir.mkdir(parents=True, exist_ok=True)
        path = models_dir / _MODEL_FILENAMES[self.model_name]
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model_name: str, models_dir: Path = MODELS_DIR) -> "LipClassifier":
        """Wczytaj wytrenowany model z pliku joblib."""
        path = models_dir / _MODEL_FILENAMES[model_name]
        if not path.exists():
            raise FileNotFoundError(f"Brak pliku modelu: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Plik {path} nie zawiera obiektu LipClassifier.")
        return obj

    @classmethod
    def is_saved(cls, model_name: str, models_dir: Path = MODELS_DIR) -> bool:
        """Sprawdź czy model jest już zapisany na dysku."""
        return (models_dir / _MODEL_FILENAMES[model_name]).exists()
