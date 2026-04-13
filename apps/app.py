"""Ujednolicona aplikacja Streamlit — Cheiloskopia (Metoda 1 + Metoda 2).

Uruchomienie:
    streamlit run apps/app.py

Nawigacja:
    ── Metoda 2: Klasyczne ML ──
        1. Eksploracja danych
        2. Preprocessing
        3. Ekstrakcja cech
        4. Trening modelu
        5. Predykcja
        6. Ewaluacja

    ── Metoda 1: Tradycyjna ──
        7. Predykcja (SSIM / ORB / Histogram)
        8. Ewaluacja (5-fold CV)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from apps.method1_pages import page_evaluation as m1_eval
from apps.method1_pages import page_prediction as m1_pred
from apps.method2_pages import (
    page_01_eksploracja,
    page_02_preprocessing,
    page_03_features,
    page_04_training,
    page_05_prediction,
    page_06_evaluation,
)
from apps.sidebar_shortcuts import render_shortcuts


def main() -> None:
    st.set_page_config(
        page_title="Cheiloskopia — Analiza biometryczna ust",
        page_icon="👄",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = {
        "Metoda 2: Klasyczne ML": [
            st.Page(
                page_01_eksploracja.render,
                title="1. Eksploracja danych",
                icon=":material/explore:",
                url_path="eksploracja",
                default=True,
            ),
            st.Page(
                page_02_preprocessing.render,
                title="2. Preprocessing",
                icon=":material/tune:",
                url_path="preprocessing",
            ),
            st.Page(
                page_03_features.render,
                title="3. Ekstrakcja cech",
                icon=":material/insights:",
                url_path="features",
            ),
            st.Page(
                page_04_training.render,
                title="4. Trening modelu",
                icon=":material/model_training:",
                url_path="training",
            ),
            st.Page(
                page_05_prediction.render,
                title="5. Predykcja",
                icon=":material/psychology:",
                url_path="prediction",
            ),
            st.Page(
                page_06_evaluation.render,
                title="6. Ewaluacja",
                icon=":material/analytics:",
                url_path="evaluation",
            ),
        ],
        "Metoda 1: Tradycyjna": [
            st.Page(
                m1_pred.render,
                title="7. Predykcja (obraz–obraz)",
                icon=":material/image_search:",
                url_path="m1_prediction",
            ),
            st.Page(
                m1_eval.render,
                title="8. Ewaluacja (CV)",
                icon=":material/bar_chart:",
                url_path="m1_evaluation",
            ),
        ],
    }

    pg = st.navigation(pages)

    with st.sidebar:
        st.title("Cheiloskopia")
        st.caption("Praca magisterska")
        st.caption("Analiza biometryczna śladów ust")
        render_shortcuts()

    pg.run()


if __name__ == "__main__":
    main()
