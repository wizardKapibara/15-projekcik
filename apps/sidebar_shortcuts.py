"""Wspólny widget skrótów do folderów/plików projektu — wyświetlany w sidebarze.

Wywołaj render_shortcuts() wewnątrz bloku `with st.sidebar:`.
Przyciski otwierają foldery/pliki w Eksploratorze Windows (os.startfile).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from src.core.config import (
    BAZA_DANYCH_DIR,
    EXPORTS_DIR,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    SPLITS_DIR,
)


def _open(path: Path) -> None:
    """Otwórz folder lub plik w Eksploratorze Windows."""
    target = path if path.exists() else path.parent
    os.startfile(str(target))


def _shortcut_btn(label: str, path: Path, help_text: str = "") -> None:
    """Jeden przycisk skrótu — otwiera path w Eksploratorze."""
    exists = path.exists()
    btn_label = label if exists else f"{label} *(brak)*"
    if st.button(btn_label, use_container_width=True, help=help_text or str(path), disabled=not exists):
        _open(path)


def _count_files(path: Path, suffix: str = "*") -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(suffix)))


def _latest_file(directory: Path, pattern: str) -> Path | None:
    files = sorted(directory.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def render_shortcuts() -> None:
    """Renderuj sekcję skrótów w sidebarze. Wywołaj wewnątrz `with st.sidebar:`."""
    st.divider()
    st.markdown("**📂 Szybki dostęp**")

    # ── DANE ──────────────────────────────────────────────────────────────────
    n_base = _count_files(BAZA_DANYCH_DIR, "*.png")
    _shortcut_btn(
        f"📸 Baza danych ({n_base} zdjęć)",
        BAZA_DANYCH_DIR,
        "Zdjęcia ust do treningu i walidacji (data/baza_danych/)",
    )

    # ── MODELE ────────────────────────────────────────────────────────────────
    n_models = _count_files(MODELS_DIR, "*.joblib")
    _shortcut_btn(
        f"🤖 Modele ({n_models} plików)",
        MODELS_DIR,
        "Wytrenowane klasyfikatory .joblib oraz pipeline scaler+PCA (models/)",
    )

    # ── WYNIKI / EXCELE ──────────────────────────────────────────────────────
    excel_pred = EXPORTS_DIR / "predykcje_szczegolowe.xlsx"
    n_pred_rows = 0
    if excel_pred.exists():
        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(excel_pred), read_only=True)
            ws = wb["Predykcja"]
            n_pred_rows = max(0, ws.max_row - 1)  # minus nagłówek
            wb.close()
        except Exception:
            pass

    _shortcut_btn(
        f"📊 Excel predykcji ({n_pred_rows} wierszy)",
        excel_pred,
        "Szczegółowy arkusz z cechami i wynikami wszystkich predykcji (exports/predykcje_szczegolowe.xlsx)",
    )

    # Ostatni plik wyników ewaluacji
    latest_result = _latest_file(RESULTS_DIR, "*.xlsx")
    if latest_result:
        _shortcut_btn(
            f"📋 Ostatnia ewaluacja",
            latest_result,
            f"Najnowszy plik wyników CV: {latest_result.name}",
        )
    else:
        st.button("📋 Ostatnia ewaluacja *(brak)*", use_container_width=True, disabled=True)

    # Foldery zbiorcze
    n_plots = _count_files(PLOTS_DIR, "*.png")
    _shortcut_btn(
        f"🖼️ Wykresy ({n_plots} plików)",
        PLOTS_DIR,
        "Macierze pomyłek i inne wykresy PNG (results/plots/)",
    )

    n_exports = _count_files(EXPORTS_DIR, "*.xlsx")
    _shortcut_btn(
        f"💾 Folder eksportów ({n_exports} plików)",
        EXPORTS_DIR,
        "Wszystkie pliki Excel wyeksportowane z aplikacji (exports/)",
    )

    # ── DANE POŚREDNIE ────────────────────────────────────────────────────────
    with st.expander("🔧 Dane pośrednie"):
        n_cache = _count_files(PROCESSED_DIR, "**/*.npy")
        _shortcut_btn(
            f"⚙️ Cache cech ({n_cache} plików .npy)",
            PROCESSED_DIR,
            "Wyekstrahowane wektory cech zapisane jako .npy (data/processed/)",
        )
        n_splits = _count_files(SPLITS_DIR, "*.json")
        _shortcut_btn(
            f"✂️ Podziały foldów ({n_splits} plików)",
            SPLITS_DIR,
            "Pliki JSON z podziałem na fold-train/test (data/splits/)",
        )
        _shortcut_btn(
            "📁 Katalog główny projektu",
            PROJECT_ROOT,
            str(PROJECT_ROOT),
        )

    st.divider()
