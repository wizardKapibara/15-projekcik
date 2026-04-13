"""Strona 2 — Pipeline preprocessingu (krok po kroku).

Kolejność kroków:
    1. Oryginał (wczytany z dysku)
    2. Skala szarości
    3. CLAHE (adaptacyjne wyrównanie histogramu)
    4. Bilateral filter (redukcja szumu, zachowanie krawędzi)
    5. Resize do 512×256

Każdy krok:
    - Podgląd obrazu PRZED i PO.
    - Suwaki parametrów.
    - Informacja o wymiarach i zakresie jasności.
Przycisk "Zapisz do cache" → data/processed/<osoba>/<obraz>/
Przycisk "Eksportuj do dokumentacji" → exports/<timestamp>/
"""

from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st

from src.core.config import BAZA_DANYCH_DIR, EXPORTS_DIR, TARGET_SIZE
from src.core.dataset_loader import LipDataset
from src.method2_ml.preprocessing import (
    export_steps_to_documentation,
    run_pipeline,
    run_pipeline_from_path,
    save_steps_to_cache,
)


@st.cache_resource
def _dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


def _img_info(img: np.ndarray) -> str:
    """Opis techniczny obrazu (wymiary, zakres pikseli)."""
    if img.ndim == 2:
        h, w = img.shape
        channels = "1-kanałowy (grayscale)"
    else:
        h, w, c = img.shape
        channels = f"{c}-kanałowy"
    return f"{w}×{h} px | {channels} | min={img.min()} max={img.max()} mean={img.mean():.1f}"


def render() -> None:
    st.title("Pipeline preprocessingu")
    st.caption(
        "Krok po kroku — każdy etap przetwarzania z regulacją parametrów. "
        "Domyślne wartości są optymalne dla biometrii ust."
    )

    try:
        dataset = _dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    persons = dataset.list_persons()

    # ----- Wybór zdjęcia -----
    st.header("Wybór zdjęcia")
    col_sel, col_btn = st.columns([3, 1])

    with col_sel:
        all_paths = dataset.all_paths()
        all_names = [p.name for p in all_paths]
        path_by_name = {p.name: p for p in all_paths}

        if "selected_image_name" not in st.session_state:
            st.session_state.selected_image_name = all_names[0]

        selected_name = st.selectbox(
            "Zdjęcie z bazy",
            all_names,
            index=all_names.index(st.session_state.selected_image_name),
            key="sel_image",
        )
        st.session_state.selected_image_name = selected_name

    with col_btn:
        st.write("")  # Wyrównanie pionowe
        st.write("")
        if st.button("🎲 Losowe", use_container_width=True):
            st.session_state.selected_image_name = random.choice(all_names)
            st.rerun()

    image_path = path_by_name[st.session_state.selected_image_name]

    # ----- Parametry suwaki -----
    st.header("Parametry preprocessingu")
    with st.expander("Dostosuj parametry (kliknij aby rozwinąć)", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("CLAHE")
            clip_limit = st.slider(
                "Clip limit (kontrast)",
                min_value=0.5, max_value=8.0, value=2.0, step=0.5,
                help="Im wyższy, tym mocniejsze wzmocnienie kontrastu. Zbyt wysokie → sztuczny wygląd."
            )
            tile_size = st.slider(
                "Tile size (rozmiar kafla)",
                min_value=4, max_value=32, value=8, step=4,
                help="Rozmiar kafla adaptacji. Mniejszy = bardziej lokalne, większy = bardziej globalne."
            )
        with col2:
            st.subheader("Bilateral filter")
            bilat_d = st.slider(
                "Diameter (d)",
                min_value=3, max_value=15, value=9, step=2,
                help="Rozmiar otoczenia piksela. Większy = wolniejszy, ale gładszy efekt."
            )
            bilat_sc = st.slider(
                "Sigma color",
                min_value=10.0, max_value=150.0, value=75.0, step=5.0,
                help="Wpływ różnicy jasności. Wyższy = więcej wygładzania ponad krawędziami."
            )
            bilat_ss = st.slider(
                "Sigma space",
                min_value=10.0, max_value=150.0, value=75.0, step=5.0,
                help="Wpływ odległości pikseli. Wyższy = bierze pod uwagę dalsze piksele."
            )

        target_w, target_h = TARGET_SIZE
        st.write(f"Docelowy rozmiar po resize: **{target_w}×{target_h} px** (stała konfiguracji)")

    # ----- Uruchom pipeline -----
    with st.spinner("Przetwarzam obraz..."):
        try:
            image_bgr = LipDataset.load_image(image_path, color="bgr")
            result = run_pipeline(
                image_bgr,
                clahe_clip_limit=clip_limit,
                clahe_tile_size=tile_size,
                bilateral_d=bilat_d,
                bilateral_sigma_color=bilat_sc,
                bilateral_sigma_space=bilat_ss,
            )
        except Exception as e:
            st.error(f"Błąd przetwarzania: {e}")
            return

    st.divider()

    # ----- Krok 1: Oryginał -----
    st.header("Krok 1 — Oryginał")
    st.caption(_img_info(result.original_rgb))
    st.image(result.original_rgb, use_container_width=True)

    st.divider()

    # ----- Krok 2: Skala szarości -----
    st.header("Krok 2 — Skala szarości")
    st.caption("Konwersja RGB → Gray (wzór: Y = 0.299R + 0.587G + 0.114B)")
    col_before, col_after = st.columns(2)
    with col_before:
        st.write("**Przed**")
        st.caption(_img_info(result.original_rgb))
        st.image(result.original_rgb, use_container_width=True)
    with col_after:
        st.write("**Po**")
        st.caption(_img_info(result.grayscale))
        st.image(result.grayscale, use_container_width=True)

    st.divider()

    # ----- Krok 3: CLAHE -----
    st.header("Krok 3 — CLAHE")
    st.caption(
        f"Adaptive Contrast Enhancement | clip_limit={result.clahe_clip_limit} | "
        f"tile={result.clahe_tile_size}×{result.clahe_tile_size}"
    )
    st.info(
        "CLAHE (Contrast Limited Adaptive Histogram Equalization) wzmacnia kontrast lokalnie "
        "w kaflach, co uwydatnia bruzdy ust bez przepalania jasnych obszarów."
    )
    col_before, col_after = st.columns(2)
    with col_before:
        st.write("**Przed (grayscale)**")
        st.caption(_img_info(result.grayscale))
        st.image(result.grayscale, use_container_width=True)
    with col_after:
        st.write("**Po CLAHE**")
        st.caption(_img_info(result.clahe))
        st.image(result.clahe, use_container_width=True)

    # Histogram porównawczy
    with st.expander("Porównanie histogramów (grayscale vs CLAHE)"):
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.hist(result.grayscale.ravel(), bins=128, color="gray", alpha=0.8)
        ax1.set_title("Histogram: grayscale")
        ax1.set_xlabel("Jasność (0-255)")
        ax1.set_ylabel("Liczba pikseli")
        ax2.hist(result.clahe.ravel(), bins=128, color="steelblue", alpha=0.8)
        ax2.set_title("Histogram: po CLAHE")
        ax2.set_xlabel("Jasność (0-255)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()

    # ----- Krok 4: Bilateral filter -----
    st.header("Krok 4 — Bilateral filter (redukcja szumu)")
    st.caption(
        f"d={result.bilateral_d} | σ_color={result.bilateral_sigma_color} | "
        f"σ_space={result.bilateral_sigma_space}"
    )
    st.info(
        "Bilateral filter wygładza obraz jak zwykłe rozmycie, ale zachowuje krawędzie — "
        "bierze pod uwagę zarówno odległość pikseli, jak i różnicę ich jasności. "
        "Chroni bruzdy ust przed zatarciem."
    )
    col_before, col_after = st.columns(2)
    with col_before:
        st.write("**Przed (CLAHE)**")
        st.caption(_img_info(result.clahe))
        st.image(result.clahe, use_container_width=True)
    with col_after:
        st.write("**Po bilateral filter**")
        st.caption(_img_info(result.denoised))
        st.image(result.denoised, use_container_width=True)

    st.divider()

    # ----- Krok 5: Resize -----
    st.header("Krok 5 — Resize")
    st.caption(f"Zmiana rozmiaru → {result.target_size[0]}×{result.target_size[1]} px (interpolacja AREA)")
    st.info(
        "Standaryzacja rozmiaru wejścia — wszystkie zdjęcia mają teraz identyczny rozmiar "
        f"{result.target_size[0]}×{result.target_size[1]} px przed ekstrakcją cech."
    )
    col_before, col_after = st.columns(2)
    with col_before:
        st.write("**Przed (bilateral)**")
        st.caption(_img_info(result.denoised))
        st.image(result.denoised, use_container_width=True)
    with col_after:
        st.write("**Po resize**")
        st.caption(_img_info(result.resized))
        st.image(result.resized, use_container_width=True)

    st.divider()

    # ----- Akcje: zapis -----
    st.header("Zapis wyników")
    col_cache, col_export = st.columns(2)

    with col_cache:
        st.subheader("Cache")
        st.caption("Zapisz etapy pośrednie do folderu data/processed/")
        if st.button("💾 Zapisz do cache", use_container_width=True):
            saved = save_steps_to_cache(result, image_path)
            st.success(f"Zapisano {len(saved)} plików:")
            for name, path in saved.items():
                st.code(str(path))

    with col_export:
        st.subheader("Dokumentacja")
        st.caption("Eksportuj z czytelnymi nazwami do folderu exports/")
        if st.button("📁 Eksportuj do dokumentacji", use_container_width=True):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_subdir = EXPORTS_DIR / f"preprocessing_{ts}"
            exported = export_steps_to_documentation(result, image_path, export_subdir)
            st.success(f"Wyeksportowano {len(exported)} plików do:")
            st.code(str(export_subdir))
            for p in exported:
                st.caption(p.name)
