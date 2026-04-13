"""Klasyfikator dla Metody 1 (tradycyjne porównanie obrazów).

1-NN po maksymalnym score podobieństwa.
Obsługuje SSIM, ORB, histogram oraz Combined (ważona średnia).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import numpy as np

from src.core.dataset_loader import LipDataset
from src.method1_traditional.histogram_compare import histogram_score
from src.method1_traditional.orb_compare import orb_score
from src.method1_traditional.ssim_compare import ssim_score

Method = Literal["ssim", "orb", "hist", "combined"]

# Wagi dla combined (muszą sumować się do 1)
COMBINED_WEIGHTS: Dict[str, float] = {
    "ssim": 0.5,
    "orb": 0.25,
    "hist": 0.25,
}


class TraditionalClassifier:
    """Klasyfikator 1-NN przez podobieństwo obrazów."""

    def __init__(self) -> None:
        self._train_paths: List[Path] = []
        self._train_labels: List[int] = []
        self._train_images: List[np.ndarray] = []  # obrazy RGB w pamięci

    def fit(self, train_paths: List[Path | str], train_labels: List[int]) -> "TraditionalClassifier":
        """Załaduj zbiór treningowy do pamięci.

        Args:
            train_paths: ścieżki do zdjęć treningowych.
            train_labels: etykiety (person_id).
            progress_callback: opcjonalne (i, n) wywoływane per obraz.
        """
        self._train_paths = [Path(p) for p in train_paths]
        self._train_labels = list(train_labels)
        self._train_images = [
            LipDataset.load_image(p, color="rgb") for p in self._train_paths
        ]
        return self

    @staticmethod
    def _pair_score(query_img: np.ndarray, train_img: np.ndarray, method: Method) -> float:
        """Oblicz pojedynczy score podobieństwa między dwoma obrazami."""
        if method == "ssim":
            return float(ssim_score(query_img, train_img))
        if method == "orb":
            return float(orb_score(query_img, train_img))
        if method == "hist":
            return float(histogram_score(query_img, train_img))
        # combined: ważona średnia (SSIM i hist normalizowane z [-1,1] → [0,1])
        s_v = ssim_score(query_img, train_img)
        o_v = orb_score(query_img, train_img)
        h_v = histogram_score(query_img, train_img)
        return float(
            COMBINED_WEIGHTS["ssim"] * (s_v + 1) / 2
            + COMBINED_WEIGHTS["orb"] * o_v
            + COMBINED_WEIGHTS["hist"] * (h_v + 1) / 2
        )

    def predict(
        self,
        query_img: np.ndarray,
        method: Method = "ssim",
        top_k: int = 5,
        progress_callback=None,
    ) -> Dict:
        """Predykcja dla jednego zdjęcia.

        Args:
            query_img: obraz RGB do identyfikacji.
            method: miara podobieństwa.
            top_k: ile kandydatów zwrócić.
            progress_callback: opcjonalnie (i, n).

        Returns:
            Słownik:
                'predicted': int — top-1 person_id,
                'confidence': float — score top-1,
                'top_k': List[Tuple[int, float, int]] — (person_id, score, index),
                'all_scores': np.ndarray — score per obraz treningowy.
        """
        if not self._train_images:
            raise RuntimeError("Model nie jest wytrenowany. Wywołaj fit().")

        n = len(self._train_images)
        scores = np.empty(n, dtype=np.float32)
        for i, train_img in enumerate(self._train_images):
            scores[i] = self._pair_score(query_img, train_img, method)
            if progress_callback:
                progress_callback(i + 1, n)

        # Top-k po malejącym score — 1-NN głosowanie
        sorted_idx = np.argsort(scores)[::-1]
        top_k_results: List[Tuple[int, float, int]] = [
            (self._train_labels[idx], float(scores[idx]), idx)
            for idx in sorted_idx[:top_k]
        ]

        return {
            "predicted": top_k_results[0][0],
            "confidence": top_k_results[0][1],
            "top_k": top_k_results,
            "all_scores": scores,
        }

    def score(
        self,
        query_paths: List[Path | str],
        query_labels: List[int],
        method: Method = "ssim",
        progress_callback=None,
    ) -> Dict:
        """Ewaluacja na zbiorze zapytań.

        Returns:
            Słownik z accuracy, top3, top5, per-query predykcjami.
        """
        correct_1 = 0
        correct_3 = 0
        correct_5 = 0
        predictions = []

        total = len(query_paths)
        for i, (path, true_label) in enumerate(zip(query_paths, query_labels)):
            img = LipDataset.load_image(path, color="rgb")
            result = self.predict(img, method=method, top_k=5)
            top5_labels = [pid for pid, _, _ in result["top_k"]]

            is_correct_1 = (result["predicted"] == true_label)
            is_correct_3 = (true_label in top5_labels[:3])
            is_correct_5 = (true_label in top5_labels)

            correct_1 += is_correct_1
            correct_3 += is_correct_3
            correct_5 += is_correct_5

            predictions.append({
                "true_id": true_label,
                "predicted": result["predicted"],
                "confidence": result["confidence"],
                "top5": top5_labels,
                "correct": is_correct_1,
            })

            if progress_callback:
                progress_callback(i + 1, total)

        return {
            "accuracy": correct_1 / total,
            "top3": correct_3 / total,
            "top5": correct_5 / total,
            "predictions": predictions,
        }
