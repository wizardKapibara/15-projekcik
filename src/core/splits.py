"""Podział datasetu na foldy cross-validation (stratified per osoba).

5-fold CV: każda osoba jest reprezentowana proporcjonalnie w każdym foldzie.
Foldy są deterministyczne (random_state=42) i zapisywane do plików JSON,
żeby były identyczne między sesjami.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.core.config import N_SPLITS, RANDOM_SEED, SPLITS_DIR
from src.core.dataset_loader import LipDataset


@dataclass
class Fold:
    """Jeden fold cross-validation."""

    fold_index: int
    train_paths: List[str]   # ścieżki jako stringi (do serializacji JSON)
    test_paths: List[str]
    train_labels: List[int]
    test_labels: List[int]

    @property
    def n_train(self) -> int:
        return len(self.train_paths)

    @property
    def n_test(self) -> int:
        return len(self.test_paths)

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "train_paths": self.train_paths,
            "test_paths": self.test_paths,
            "train_labels": self.train_labels,
            "test_labels": self.test_labels,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Fold":
        return cls(
            fold_index=d["fold_index"],
            train_paths=d["train_paths"],
            test_paths=d["test_paths"],
            train_labels=d["train_labels"],
            test_labels=d["test_labels"],
        )


def make_5fold_splits(
    dataset: LipDataset,
    n_splits: int = N_SPLITS,
    random_state: int = RANDOM_SEED,
) -> List[Fold]:
    """Utwórz n_splits foldów stratified CV.

    Args:
        dataset: załadowany LipDataset.
        n_splits: liczba foldów (domyślnie 5).
        random_state: seed dla reprodukowalności.

    Returns:
        Lista obiektów Fold.
    """
    all_pairs = dataset.all_pairs()
    all_paths = [str(path) for path, _ in all_pairs]
    all_labels = [parsed.person_id for _, parsed in all_pairs]

    X = np.array(all_paths)
    y = np.array(all_labels)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    folds: List[Fold] = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        folds.append(Fold(
            fold_index=fold_idx,
            train_paths=X[train_idx].tolist(),
            test_paths=X[test_idx].tolist(),
            train_labels=y[train_idx].tolist(),
            test_labels=y[test_idx].tolist(),
        ))

    return folds


def save_splits(folds: List[Fold], splits_dir: Path = SPLITS_DIR) -> List[Path]:
    """Zapisz foldy do plików JSON."""
    splits_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []
    for fold in folds:
        path = splits_dir / f"fold_{fold.fold_index}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(fold.to_dict(), f, indent=2, ensure_ascii=False)
        saved.append(path)
    return saved


def load_splits(splits_dir: Path = SPLITS_DIR) -> List[Fold]:
    """Wczytaj zapisane foldy z plików JSON."""
    folds: List[Fold] = []
    for i in range(N_SPLITS):
        path = splits_dir / f"fold_{i}.json"
        if not path.exists():
            raise FileNotFoundError(f"Brak pliku folda: {path}")
        with open(path, "r", encoding="utf-8") as f:
            folds.append(Fold.from_dict(json.load(f)))
    return sorted(folds, key=lambda f: f.fold_index)


def splits_exist(splits_dir: Path = SPLITS_DIR) -> bool:
    """Sprawdź czy wszystkie pliki foldów istnieją."""
    return all((splits_dir / f"fold_{i}.json").exists() for i in range(N_SPLITS))


def get_or_create_splits(dataset: LipDataset) -> List[Fold]:
    """Wczytaj istniejące foldy lub utwórz i zapisz nowe."""
    if splits_exist():
        return load_splits()
    folds = make_5fold_splits(dataset)
    save_splits(folds)
    return folds
