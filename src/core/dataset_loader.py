"""Loader bazy zdjęć ust z folderu data/baza_danych.

Skanuje folder, parsuje nazwy plików (filename_parser), grupuje po osobach.
Ładuje obrazy w sposób bezpieczny dla Windows i ścieżek z polskimi znakami / spacjami.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from .filename_parser import ParsedFilename, try_parse_filename


class LipDataset:
    """Dataset zdjęć ust pogrupowanych po osobach."""

    def __init__(self, root: Path | str) -> None:
        """
        Args:
            root: ścieżka do folderu z plikami PNG (np. data/baza_danych).
        """
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Folder bazy danych nie istnieje: {self.root}")
        if not self.root.is_dir():
            raise NotADirectoryError(f"To nie jest folder: {self.root}")

        self._index: Dict[int, List[Tuple[Path, ParsedFilename]]] = {}
        self._all_paths: List[Path] = []
        self._scan()

    def _scan(self) -> None:
        """Zeskanuj folder i zbuduj indeks osoba → lista plików."""
        for path in sorted(self.root.iterdir()):
            if not path.is_file():
                continue
            parsed = try_parse_filename(path)
            if parsed is None:
                continue
            self._index.setdefault(parsed.person_id, []).append((path, parsed))
            self._all_paths.append(path)

        # Sortujemy zdjęcia w obrębie każdej osoby po dacie/czasie dla determinizmu
        for person_id in self._index:
            self._index[person_id].sort(key=lambda x: (x[1].date, x[1].time))

    # ----- API listowania -----

    def list_persons(self) -> List[int]:
        """Zwraca posortowaną listę numerów osób w bazie."""
        return sorted(self._index.keys())

    def list_images(self, person_id: int) -> List[Path]:
        """Zwraca listę ścieżek zdjęć danej osoby."""
        if person_id not in self._index:
            return []
        return [path for path, _ in self._index[person_id]]

    def list_metadata(self, person_id: int) -> List[ParsedFilename]:
        """Zwraca listę sparsowanych metadanych zdjęć danej osoby."""
        if person_id not in self._index:
            return []
        return [parsed for _, parsed in self._index[person_id]]

    def all_paths(self) -> List[Path]:
        """Zwraca listę wszystkich ścieżek zdjęć w bazie."""
        return list(self._all_paths)

    def all_pairs(self) -> List[Tuple[Path, ParsedFilename]]:
        """Zwraca listę (ścieżka, metadane) dla wszystkich zdjęć."""
        return [pair for person_id in self.list_persons() for pair in self._index[person_id]]

    def count_per_person(self) -> Dict[int, int]:
        """Zwraca słownik person_id → liczba zdjęć."""
        return {pid: len(items) for pid, items in self._index.items()}

    def total_count(self) -> int:
        """Łączna liczba zdjęć w bazie."""
        return len(self._all_paths)

    # ----- API ładowania obrazów -----

    @staticmethod
    def load_image(
        path: Path | str,
        color: str = "rgb",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Załaduj obraz z dysku.

        Args:
            path: ścieżka do pliku obrazu.
            color: 'rgb', 'bgr' lub 'gray'.
            target_size: (width, height) — opcjonalnie, zmiana rozmiaru.

        Returns:
            np.ndarray z obrazem.

        Raises:
            FileNotFoundError: jeśli plik nie istnieje.
            ValueError: jeśli nie udało się odczytać obrazu.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Plik nie istnieje: {path}")

        # np.fromfile + cv2.imdecode jest bezpieczne dla ścieżek
        # z Unicode/spacjami na Windows (cv2.imread bywa zawodne).
        raw = np.fromfile(str(path), dtype=np.uint8)
        if raw.size == 0:
            raise ValueError(f"Pusty plik: {path}")

        if color == "gray":
            img = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if img is not None and color == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            raise ValueError(f"Nie udało się zdekodować obrazu: {path}")

        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        return img

    def iterate_all(
        self,
        color: str = "rgb",
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Iterator[Tuple[int, Path, np.ndarray]]:
        """Iteruje przez wszystkie zdjęcia: yield (person_id, path, image)."""
        for person_id in self.list_persons():
            for path, _ in self._index[person_id]:
                yield person_id, path, self.load_image(path, color=color, target_size=target_size)
