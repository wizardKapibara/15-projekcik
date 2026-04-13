"""Parser nazw plików zdjęć ust z bazy danych.

Format nazwy pliku:
    YYYY-MM-DD_HH-MM-SS_Nr_X_LampaON|OFF_Fokus_A_B_Exp_E_CUT.png

Przykłady:
    2026-01-06_20-54-12_Nr_01_LampaON_Fokus_1_0_Exp_0_CUT.png
    2026-03-05_08-25-34_Nr_2_LampaOFF_Fokus_0_68_Exp_0_CUT.png
    2026-03-05_08-30-46_Nr_3_LampaON_Fokus_0_89_Exp_-2_CUT.png

Uwaga: numer osoby może być z zerem wiodącym (Nr_01) lub bez (Nr_2).
Parser normalizuje do zwykłego inta.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Union


_FILENAME_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})"
    r"_(?P<time>\d{2}-\d{2}-\d{2})"
    r"_Nr_(?P<person>\d+)"
    r"_Lampa(?P<flash>ON|OFF)"
    r"_Fokus_(?P<focus_int>\d+)_(?P<focus_frac>\d+)"
    r"_Exp_(?P<exposure>-?\d+)"
    r"_CUT"
    r"\.(?P<ext>png|jpg|jpeg)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedFilename:
    """Rozbita nazwa pliku zdjęcia ust."""

    date: date
    time: time
    person_id: int  # znormalizowany int (1-22), niezależnie od zera wiodącego
    flash: str  # "ON" lub "OFF"
    focus: float  # np. 0.68, 1.0
    exposure: int  # może być ujemne, np. -4
    is_cut: bool  # zawsze True jeśli spełnia wzorzec (sufiks _CUT)
    extension: str  # "png", "jpg", "jpeg"
    original_name: str  # oryginalna nazwa pliku

    @property
    def person_label(self) -> str:
        """Etykieta tekstowa z zerem wiodącym, np. 'Nr_05'."""
        return f"Nr_{self.person_id:02d}"


def parse_filename(name: Union[str, Path]) -> ParsedFilename:
    """Sparsuj nazwę pliku zdjęcia ust.

    Args:
        name: nazwa pliku (string lub Path; wystarczy sama nazwa, ścieżka też OK).

    Returns:
        ParsedFilename z polami opisującymi metadane.

    Raises:
        ValueError: jeśli nazwa nie pasuje do oczekiwanego wzorca.
    """
    if isinstance(name, Path):
        filename = name.name
    else:
        filename = Path(name).name

    match = _FILENAME_RE.match(filename)
    if not match:
        raise ValueError(f"Nazwa pliku nie pasuje do wzorca: {filename!r}")

    groups = match.groupdict()

    year, month, day = map(int, groups["date"].split("-"))
    hour, minute, second = map(int, groups["time"].split("-"))

    focus_value = float(f"{groups['focus_int']}.{groups['focus_frac']}")

    return ParsedFilename(
        date=date(year, month, day),
        time=time(hour, minute, second),
        person_id=int(groups["person"]),
        flash=groups["flash"].upper(),
        focus=focus_value,
        exposure=int(groups["exposure"]),
        is_cut=True,
        extension=groups["ext"].lower(),
        original_name=filename,
    )


def try_parse_filename(name: Union[str, Path]) -> ParsedFilename | None:
    """Jak parse_filename, ale zwraca None zamiast podnosić wyjątek."""
    try:
        return parse_filename(name)
    except ValueError:
        return None
