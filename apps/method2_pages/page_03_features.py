"""Strona 3 — Ekstrakcja cech biometrycznych (LBP, HOG, Gabor, minucje).

Dla wybranego zdjęcia pokazuje każdy typ cechy z wizualizacją i opisem.
"""

from __future__ import annotations

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.core.config import BAZA_DANYCH_DIR
from src.core.dataset_loader import LipDataset
from src.method2_ml.features_gabor import extract_gabor, gabor_responses_grid
from src.method2_ml.features_hog import extract_hog
from src.method2_ml.features_lbp import extract_lbp
from src.method2_ml.features_minutiae import draw_minutiae, extract_minutiae
from src.method2_ml.preprocessing import run_pipeline_from_path


@st.cache_resource
def _dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


@st.cache_data(show_spinner=False)
def _run_preprocessing(image_path_str: str) -> dict:
    """Cache preprocessingu — uruchamiamy raz per zdjęcie."""
    result = run_pipeline_from_path(image_path_str)
    return {
        "original_rgb": result.original_rgb,
        "grayscale": result.grayscale,
        "clahe": result.clahe,
        "denoised": result.denoised,
        "resized": result.resized,
    }


def render() -> None:
    st.title("Ekstrakcja cech biometrycznych")
    st.caption(
        "Każda cecha opisuje inny aspekt bruzd ust. "
        "Razem tworzą wektor cech przekazywany klasyfikatorowi."
    )

    try:
        dataset = _dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ----- Wybór zdjęcia -----
    st.header("Wybór zdjęcia")
    all_paths = dataset.all_paths()
    all_names = [p.name for p in all_paths]
    path_by_name = {p.name: p for p in all_paths}

    if "feat_image_name" not in st.session_state:
        st.session_state.feat_image_name = all_names[0]

    col_sel, col_rnd = st.columns([4, 1])
    with col_sel:
        selected_name = st.selectbox(
            "Zdjęcie", all_names,
            index=all_names.index(st.session_state.feat_image_name),
            key="feat_sel",
        )
        st.session_state.feat_image_name = selected_name
    with col_rnd:
        st.write("")
        st.write("")
        if st.button("🎲 Losowe", use_container_width=True, key="feat_rnd"):
            st.session_state.feat_image_name = random.choice(all_names)
            st.rerun()

    image_path = path_by_name[st.session_state.feat_image_name]

    # Preprocessing (z cache)
    with st.spinner("Preprocessing..."):
        prep = _run_preprocessing(str(image_path))
    resized = prep["resized"]

    st.image(prep["original_rgb"], caption="Oryginał", use_container_width=True)
    st.caption(f"Wejście do ekstrakcji cech: obraz po preprocessingu {resized.shape[1]}×{resized.shape[0]} px")
    st.divider()

    # ===============================================
    # 1. LBP — Local Binary Patterns
    # ===============================================
    st.header("1. LBP — Local Binary Patterns")
    st.info(
        "LBP koduje lokalną teksturę: dla każdego piksela porównuje go z sąsiadami "
        "i zapisuje wynik jako kod binarny. Histogram tych kodów opisuje wzorce tekstury bruzd. "
        "**Zaleta**: szybki, odporny na monotoniczne zmiany jasności."
    )

    with st.expander("Parametry LBP"):
        col1, col2 = st.columns(2)
        lbp_p = col1.slider("P (liczba sąsiadów)", 4, 24, 8, 4)
        lbp_r = col2.slider("R (promień)", 1, 4, 1)
        lbp_bins = st.slider("Liczba binów histogramu", 8, 32, 10, 2)

    lbp_img, lbp_hist = extract_lbp(resized, n_points=lbp_p, radius=lbp_r, n_bins=lbp_bins)

    col_lbp_img, col_lbp_hist = st.columns(2)
    with col_lbp_img:
        st.write("**Mapa wzorców LBP**")
        # Normalizuj do wyświetlenia
        lbp_vis = ((lbp_img / lbp_img.max()) * 255).astype(np.uint8) if lbp_img.max() > 0 else lbp_img.astype(np.uint8)
        st.image(lbp_vis, use_container_width=True)
    with col_lbp_hist:
        st.write("**Histogram LBP (wektor cech)**")
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(len(lbp_hist)), lbp_hist, color="steelblue")
        ax.set_xlabel("Wzorzec LBP")
        ax.set_ylabel("Częstość (znorm.)")
        ax.set_title(f"LBP_{{P={lbp_p}, R={lbp_r}}} — {len(lbp_hist)} cech")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.success(f"Wektor cech LBP: **{len(lbp_hist)} wartości** | {lbp_hist.round(4)}")
    st.divider()

    # ===============================================
    # 2. HOG — Histogram of Oriented Gradients
    # ===============================================
    st.header("2. HOG — Histogram of Oriented Gradients")
    st.info(
        "HOG dzieli obraz na siatkę komórek i w każdej oblicza histogram kierunków krawędzi. "
        "Doskonale rozróżnia bruzdy pionowe, poziome i ukośne. "
        "**Zaleta**: bardzo bogaty opis struktury liniowej obrazu."
    )

    with st.expander("Parametry HOG"):
        col1, col2, col3 = st.columns(3)
        hog_orient = col1.slider("Orientacje", 6, 16, 9, 1)
        hog_ppc = col2.slider("Piksele/komórkę", 8, 32, 16, 8)
        hog_cpb = col3.slider("Komórki/blok", 1, 4, 2, 1)

    hog_vis, hog_fv = extract_hog(resized, orientations=hog_orient,
                                   pixels_per_cell=(hog_ppc, hog_ppc),
                                   cells_per_block=(hog_cpb, hog_cpb))

    col_hog_img, col_hog_info = st.columns(2)
    with col_hog_img:
        st.write("**Wizualizacja gradientów HOG**")
        hog_vis_uint8 = ((hog_vis / hog_vis.max()) * 255).astype(np.uint8) if hog_vis.max() > 0 else hog_vis.astype(np.uint8)
        st.image(hog_vis_uint8, use_container_width=True)
    with col_hog_info:
        st.write("**Informacje o wektorze**")
        st.metric("Długość wektora HOG", len(hog_fv))
        st.metric("Min wartość", f"{hog_fv.min():.4f}")
        st.metric("Max wartość", f"{hog_fv.max():.4f}")
        st.metric("Średnia", f"{hog_fv.mean():.4f}")
        st.caption(
            f"Wymiary: {resized.shape[1]}×{resized.shape[0]} px, "
            f"komórki: {hog_ppc}×{hog_ppc}, "
            f"bloki: {hog_cpb}×{hog_cpb}, "
            f"orientacje: {hog_orient}"
        )

    st.success(f"Wektor cech HOG: **{len(hog_fv)} wartości**")
    st.divider()

    # ===============================================
    # 3. Gabor — bank filtrów kierunkowych
    # ===============================================
    st.header("3. Filtry Gabora — bank kierunkowy")
    st.info(
        "Filtry Gabora modelują percepcję wzrokową kory mózgowej. "
        "Każdy filtr reaguje na bruzdy o konkretnej orientacji i częstotliwości. "
        "Odpowiedź filtra = jak mocno ten kierunek bruzd jest obecny w obrazie. "
        "**Zaleta**: najlepsze dla tekstur o charakterze falistym/liniowym (bruzdy ust)."
    )

    with st.expander("Parametry Gabora"):
        col1, col2 = st.columns(2)
        gab_orient = col1.slider("Liczba orientacji", 4, 12, 8, 2)
        gab_freqs_str = col2.multiselect(
            "Częstotliwości",
            [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
            default=[0.1, 0.2, 0.3, 0.4],
        )
    gab_freqs = tuple(sorted(gab_freqs_str)) if gab_freqs_str else (0.1, 0.2, 0.3, 0.4)

    with st.spinner("Obliczam odpowiedzi filtrów Gabora..."):
        gab_responses, gab_fv = extract_gabor(resized, orientations=gab_orient, frequencies=gab_freqs)

    st.write("**Siatka odpowiedzi filtrów** (wiersze = częstotliwości, kolumny = orientacje)")
    grid = gabor_responses_grid(gab_responses, orientations=gab_orient)
    st.image(grid, use_container_width=True)
    st.success(
        f"Bank: {len(gab_freqs)} częstotliwości × {gab_orient} orientacji = "
        f"{len(gab_responses)} filtrów → **{len(gab_fv)} cech** (mean+std per filtr)"
    )
    st.divider()

    # ===============================================
    # 4. Minucje — cheiloskopijne punkty charakterystyczne
    # ===============================================
    st.header("4. Minucje — punkty charakterystyczne bruzd")
    st.info(
        "Minucje to punkty, gdzie bruzda się zaczyna/kończy, rozwidla lub przecina. "
        "Metoda: binaryzacja → skeletonizacja → Crossing Number (CN). "
        "**CN=1**: zakończenie (zielony) | **CN=3**: rozwidlenie (czerwony) | **CN≥4**: skrzyżowanie (niebieski)"
    )

    with st.expander("Parametry minucji"):
        col1, col2, col3 = st.columns(3)
        bin_method = col1.selectbox(
            "Metoda binaryzacji",
            ["adaptive", "otsu", "gabor_otsu"],
            index=0,
            help="adaptive: najlepsza dla zmiennego oświetlenia\notsu: szybsza\ngabor_otsu: najlepsza jakość"
        )
        block_size = col2.slider("Block size (adaptive)", 11, 71, 35, 2,
                                  help="Rozmiar bloku adaptacyjnego progu. Musi być nieparzysty.")
        c_const = col3.slider("C constant (adaptive)", 2, 30, 10,
                               help="Stała odejmowana od średniej bloku.")

    with st.spinner("Obliczam minucje (skeletonizacja może chwilę potrwać)..."):
        try:
            min_result = extract_minutiae(
                resized,
                binarize_method=bin_method,
                block_size=block_size if block_size % 2 == 1 else block_size + 1,
                c_constant=c_const,
            )
        except Exception as e:
            st.error(f"Błąd ekstrakcji minucji: {e}")
            return

    # Wyświetlenie kroków
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.write("**a) Oryginał (preprocessed)**")
        st.image(resized, use_container_width=True)
    with col2:
        st.write("**b) Binaryzacja**")
        st.image(min_result.binary, use_container_width=True)
    with col3:
        st.write("**c) Szkielet**")
        st.image(min_result.skeleton, use_container_width=True)
    with col4:
        st.write("**d) Minucje**")
        vis = draw_minutiae(min_result.skeleton, min_result.minutiae, radius=3)
        st.image(vis, use_container_width=True)

    # Legenda
    col_e, col_b, col_c = st.columns(3)
    col_e.markdown("🟢 **Zakończenia** (endings)")
    col_b.markdown("🔴 **Rozwidlenia** (bifurcations)")
    col_c.markdown("🔵 **Skrzyżowania** (crossings)")

    # Statystyki — wektor cech
    st.subheader("Wektor cech minucji")
    fv = min_result.feature_vector()
    feat_df = pd.DataFrame({
        "Cecha": min_result.feature_names,
        "Wartość": [
            min_result.n_endings,
            min_result.n_bifurcations,
            min_result.n_crossings,
            min_result.n_total_minutiae,
            f"{min_result.groove_density:.2f}",
            min_result.total_skeleton_pixels,
            f"{min_result.binary_entropy:.4f}",
        ],
        "Opis": [
            "Liczba zakończeń linii",
            "Liczba rozwidleń",
            "Liczba skrzyżowań (CN≥4)",
            "Łączna liczba minucji",
            "Minucji / 10 000 px szkieletu",
            "Całkowita długość bruzd [px]",
            "Entropia Shannona (złożoność wzorca)",
        ],
    })

    cols_metric = st.columns(4)
    cols_metric[0].metric("Zakończenia", min_result.n_endings)
    cols_metric[1].metric("Rozwidlenia", min_result.n_bifurcations)
    cols_metric[2].metric("Skrzyżowania", min_result.n_crossings)
    cols_metric[3].metric("Razem minucji", min_result.n_total_minutiae)

    st.dataframe(feat_df, use_container_width=True, hide_index=True)
    st.success(f"Wektor cech minucji: **{len(fv)} wartości**")

    st.divider()

    # ===============================================
    # Podsumowanie: kompletny wektor cech
    # ===============================================
    st.header("Podsumowanie — kompletny wektor cech")

    total_features = len(lbp_hist) + len(hog_fv) + len(gab_fv) + len(fv)
    summary_df = pd.DataFrame({
        "Ekstraktor": ["LBP", "HOG", "Gabor", "Minucje", "ŁĄCZNIE"],
        "Liczba cech": [len(lbp_hist), len(hog_fv), len(gab_fv), len(fv), total_features],
        "Opis": [
            "Tekstura lokalna — wzorce sąsiedztwa pikseli",
            "Kierunki krawędzi — orientacja bruzd",
            "Odpowiedzi filtrów kierunkowych (mean+std)",
            "Punkty charakterystyczne szkieletu",
            "→ Wejście do SVM / Random Forest / k-NN",
        ],
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.metric("Łączna długość wektora cech", total_features)
    st.caption(
        "Przed trenowaniem klasyfikatora wektor zostanie zestandaryzowany "
        "(StandardScaler: mean=0, std=1 per cecha)."
    )
