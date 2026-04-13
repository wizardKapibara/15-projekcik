"""Hash zawartości folderu datasetu — do wykrywania nowych/zmienionych zdjęć.

Mechanizm: SHA-256 z posortowanej listy (filename + file_size + mtime).
Lekki i wystarczająco unikalny — nie wymaga czytania treści plików.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from src.core.config import BAZA_DANYCH_DIR, MODELS_DIR

_HASH_FILE = MODELS_DIR / "dataset_hash.txt"


def compute_dataset_hash(folder: Path | str = BAZA_DANYCH_DIR) -> str:
    """Oblicz hash stanu folderu z danymi.

    Args:
        folder: folder z plikami PNG.

    Returns:
        Hex string SHA-256.
    """
    folder = Path(folder)
    entries: list[str] = []
    for p in sorted(folder.iterdir()):
        if p.is_file():
            stat = p.stat()
            entries.append(f"{p.name}:{stat.st_size}:{stat.st_mtime:.3f}")

    combined = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(combined).hexdigest()


def save_current_hash(hash_value: str, hash_file: Path = _HASH_FILE) -> None:
    """Zapisz aktualny hash do pliku po trenowaniu modelu."""
    hash_file.parent.mkdir(parents=True, exist_ok=True)
    hash_file.write_text(hash_value, encoding="utf-8")


def load_saved_hash(hash_file: Path = _HASH_FILE) -> str | None:
    """Wczytaj ostatnio zapisany hash (None jeśli nie istnieje)."""
    if not hash_file.exists():
        return None
    return hash_file.read_text(encoding="utf-8").strip()


def is_retrain_needed(
    folder: Path | str = BAZA_DANYCH_DIR,
    hash_file: Path = _HASH_FILE,
) -> bool:
    """Sprawdź czy dataset zmienił się od ostatniego trenowania.

    Returns:
        True jeśli modele wymagają ponownego trenowania.
    """
    saved = load_saved_hash(hash_file)
    if saved is None:
        return True  # Brak pliku hash = nigdy nie trenowano
    current = compute_dataset_hash(folder)
    return current != saved
