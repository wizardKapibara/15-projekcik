"""Strona 5 — Predykcja pojedynczego zdjęcia.

Uruchamia wszystkie 3 klasyfikatory (SVM / RF / k-NN) jednocześnie
i porównuje ich wyniki obok siebie.

Tryby wyboru zdjęcia:
    1. Losowe z bazy
    2. Wybór ręczny z dropdownu
    3. Upload zewnętrznego pliku
"""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.core.config import BAZA_DANYCH_DIR, METADATA_XLSM
from src.core.dataset_loader import LipDataset
from src.core.filename_parser import try_parse_filename
from src.core.metadata_loader import load_person_metadata
from src.core.results_writer import write_detailed_prediction_excel
from src.method2_ml.classifier import AVAILABLE_CLASSIFIERS, LipClassifier
from src.method2_ml.feature_vector import FeatureExtractorPipeline, extract_raw_features
from src.method2_ml.preprocessing import run_pipeline


@st.cache_resource
def _dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


@st.cache_resource
def _metadata() -> dict:
    if not METADATA_XLSM.exists():
        return {}
    return load_person_metadata(METADATA_XLSM)


def _load_pipeline() -> FeatureExtractorPipeline | None:
    try:
        return FeatureExtractorPipeline.load()
    except FileNotFoundError:
        return None


def _load_all_classifiers() -> dict[str, LipClassifier | None]:
    result = {}
    for name in AVAILABLE_CLASSIFIERS:
        try:
            result[name] = LipClassifier.load(name)
        except FileNotFoundError:
            result[name] = None
    return result


def _meta_card(person_id: int, metadata: dict) -> None:
    meta = metadata.get(person_id)
    if meta is None:
        st.caption("Brak metadanych.")
        return
    d = meta.to_dict()
    for k, v in d.items():
        if k == "person_id" or v is None:
            continue
        st.caption(f"**{k}**: {v}")


def _top5_chart(top5: list, clf_name: str, true_id: int | None) -> None:
    labels = [f"Nr_{pid:02d}" for pid, _ in top5]
    values = [conf for _, conf in top5]
    colors = []
    for pid, _ in top5:
        if true_id is not None and pid == true_id:
            colors.append("#00B050")   # zielony = prawdziwa
        elif pid == top5[0][0]:
            colors.append("#2E75B6")   # niebieski = top-1 (predicted)
        else:
            colors.append("#A9C4E2")   # szary = reszta

    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Pewność")
    ax.set_xlim(0, max(values) * 1.2 if values else 1.0)
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center", fontsize=8)
    ax.set_title(clf_name.upper(), fontsize=10, fontweight="bold")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def render() -> None:
    st.title("Predykcja pojedynczego zdjęcia")
    st.caption(
        "Zdjęcie jest analizowane przez wszystkie 3 klasyfikatory jednocześnie. "
        "Wyniki możesz zapisać do Excela ze szczegółowymi danymi o cechach i predykcji."
    )

    try:
        dataset = _dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    metadata = _metadata()

    # ---- Wczytaj modele ----
    pipeline = _load_pipeline()
    clfs = _load_all_classifiers()

    missing = [n for n, c in clfs.items() if c is None]
    if pipeline is None or missing:
        what = "pipeline" if pipeline is None else ", ".join(m.upper() for m in missing)
        st.error(
            f"Brak modelu: **{what}**. "
            "Przejdź do strony **Trening modelu** i wytrenuj klasyfikatory."
        )
        return

    st.success("✅ Wszystkie modele wczytane: SVM · RF · k-NN")
    st.divider()

    # ---- Wybór zdjęcia ----
    st.header("Wybór zdjęcia")
    all_paths = dataset.all_paths()
    all_names = [p.name for p in all_paths]
    path_by_name = {p.name: p for p in all_paths}

    tab1, tab2, tab3 = st.tabs(["🎲 Losowe z bazy", "📋 Wybór ręczny", "📤 Upload pliku"])

    selected_path: Path | None = None
    selected_is_external: bool = False
    selected_uploaded_data: bytes | None = None

    with tab1:
        if "pred_random_name" not in st.session_state:
            st.session_state.pred_random_name = random.choice(all_names)
        if st.button("🎲 Wylosuj zdjęcie", use_container_width=True):
            st.session_state.pred_random_name = random.choice(all_names)
        st.write(f"Wylosowane: `{st.session_state.pred_random_name}`")
        if st.button("Użyj losowego zdjęcia →", type="primary", key="use_random"):
            selected_path = path_by_name[st.session_state.pred_random_name]
            selected_is_external = False
            selected_uploaded_data = None

    with tab2:
        col_sel, col_use = st.columns([4, 1])
        with col_sel:
            manual_name = st.selectbox("Zdjęcie z bazy", all_names, key="pred_manual_sel")
        with col_use:
            st.write("")
            st.write("")
            if st.button("Użyj →", key="use_manual"):
                selected_path = path_by_name[manual_name]
                selected_is_external = False
                selected_uploaded_data = None

    with tab3:
        uploaded = st.file_uploader(
            "Załaduj zdjęcie spoza bazy",
            type=["png", "jpg", "jpeg"],
            help="Zdjęcie ust wycięte z twarzy, zbliżony format do bazy (panoramiczne, jasne tło)."
        )
        if uploaded is not None:
            if st.button("Użyj uploadowanego →", type="primary", key="use_upload"):
                selected_path = Path(uploaded.name)
                selected_is_external = True
                selected_uploaded_data = uploaded.getvalue()

    # ---- Session state ----
    st.divider()
    for key, default in [
        ("pred_selected_path", None),
        ("pred_is_external", False),
        ("pred_uploaded_data", None),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    if selected_path is not None:
        st.session_state.pred_selected_path = selected_path
        st.session_state.pred_is_external = selected_is_external
        st.session_state.pred_uploaded_data = selected_uploaded_data

    current_path: Path | None = st.session_state.pred_selected_path
    current_is_external: bool = st.session_state.pred_is_external
    current_uploaded: bytes | None = st.session_state.pred_uploaded_data

    if current_path is None:
        st.info("Wybierz zdjęcie z jednej z zakładek powyżej i kliknij 'Użyj →'.")
        return

    st.subheader(f"Zdjęcie: `{current_path.name}`")

    # ---- Wczytaj obraz ----
    try:
        if current_is_external and current_uploaded is not None:
            arr = np.frombuffer(current_uploaded, dtype=np.uint8)
            image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        else:
            image_bgr = LipDataset.load_image(current_path, color="bgr")
    except Exception as e:
        st.error(f"Nie można wczytać obrazu: {e}")
        return

    st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Oryginał", use_container_width=True)

    # ---- Preprocessing + cechy + predykcja (wszystkie 3 clf) ----
    with st.spinner("Preprocessing · Ekstrakcja cech · Predykcja (SVM + RF + k-NN)..."):
        t0 = time.time()

        result = run_pipeline(image_bgr)

        # Ekstrakcja cech
        if not current_is_external:
            fv = pipeline.transform_single(current_path)
        else:
            raw_feats = extract_raw_features(result.resized)
            raw_vec = np.concatenate(
                [raw_feats["lbp"], raw_feats["hog"], raw_feats["gabor"], raw_feats["minutiae"]]
            ).reshape(1, -1).astype(np.float32)
            fv = pipeline._scaler.transform(pipeline._apply_pca_and_concat(raw_vec))[0]

        # Zawsze pobierz też surowe cechy (do Excela i minucji)
        raw_feats = extract_raw_features(result.resized)

        # Predykcja wszystkimi 3 klasyfikatorami
        predictions: dict[str, dict] = {}
        for name in AVAILABLE_CLASSIFIERS:
            predictions[name] = clfs[name].predict_single(fv, top_k=5)

        t1 = time.time()
        time_total_ms = (t1 - t0) * 1000

    # ---- Prawdziwa etykieta ----
    true_id: int | None = None
    if not current_is_external:
        parsed = try_parse_filename(current_path)
        if parsed:
            true_id = parsed.person_id

    # ---- Wyniki — 3 kolumny ----
    st.header("Wyniki predykcji")
    st.caption(f"Czas przetwarzania: **{time_total_ms:.0f} ms** łącznie")

    if true_id is not None:
        st.info(f"Prawdziwa osoba: **Nr_{true_id:02d}**")

    cols = st.columns(3)
    all_correct = []

    for col, clf_name in zip(cols, AVAILABLE_CLASSIFIERS):
        pred = predictions[clf_name]
        top1_id = pred["predicted"]
        conf = pred["confidence"]
        top5 = pred["top_k"]
        is_correct = (top1_id == true_id) if true_id is not None else None

        with col:
            st.subheader(clf_name.upper())
            if is_correct is True:
                st.success(f"✅ Nr_{top1_id:02d}  ({conf:.1%})")
            elif is_correct is False:
                st.error(f"❌ Nr_{top1_id:02d}  ({conf:.1%})")
            else:
                st.metric("Predykcja", f"Nr_{top1_id:02d}", delta=f"{conf:.1%}")

            _top5_chart(top5, clf_name, true_id)
            all_correct.append(is_correct)

    # Konsensus
    tops = [predictions[n]["predicted"] for n in AVAILABLE_CLASSIFIERS]
    if len(set(tops)) == 1:
        st.success(f"✅ **Wszystkie 3 klasyfikatory zgodne:** Nr_{tops[0]:02d}")
    else:
        from collections import Counter
        cnt = Counter(tops)
        most_id, most_n = cnt.most_common(1)[0]
        if most_n >= 2:
            agreeing = [n.upper() for n in AVAILABLE_CLASSIFIERS if predictions[n]["predicted"] == most_id]
            st.warning(
                f"⚠️ Klasyfikatory **niezgodne**. "
                f"{', '.join(agreeing)} wskazują Nr_{most_id:02d}; "
                f"pozostały wskazuje Nr_{[n for n in AVAILABLE_CLASSIFIERS if predictions[n]['predicted'] != most_id][0]:02d}."
            )
        else:
            st.error("❌ Wszystkie 3 klasyfikatory wskazują różne osoby.")

    # ---- Szczegóły preprocessing + cechy ----
    st.divider()
    with st.expander("Podgląd kroków preprocessingu"):
        img_cols = st.columns(5)
        steps = [
            ("Oryginał", cv2.cvtColor(result.original_bgr, cv2.COLOR_BGR2RGB)),
            ("Grayscale", result.grayscale),
            ("CLAHE", result.clahe),
            ("Bilateral", result.denoised),
            ("Resize", result.resized),
        ]
        for c, (name, img) in zip(img_cols, steps):
            c.image(img, caption=name, use_container_width=True)

    with st.expander("Cechy minucji"):
        mv = raw_feats["minutiae"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Zakończenia", int(round(float(mv[0]))))
        c2.metric("Rozwidlenia", int(round(float(mv[1]))))
        c3.metric("Skrzyżowania", int(round(float(mv[2]))))
        c4.metric("Łącznie minucji", int(round(float(mv[3]))))
        st.caption(
            f"Gęstość bruzd: {float(mv[4]):.4f} | "
            f"Piksele szkieletu: {int(round(float(mv[5])))} | "
            f"Entropia binarna: {float(mv[6]):.4f}"
        )

    with st.expander("Metadane kandydatów"):
        meta_cols = st.columns(3)
        for mc, clf_name in zip(meta_cols, AVAILABLE_CLASSIFIERS):
            with mc:
                st.write(f"**{clf_name.upper()} → Nr_{predictions[clf_name]['predicted']:02d}**")
                _meta_card(predictions[clf_name]["predicted"], metadata)

    # ---- Zapis do Excela ----
    st.divider()
    st.subheader("Zapis do Excela")
    st.caption(
        "Zapisuje pełny wiersz z danymi preprocessingu, wszystkimi cechami "
        "(LBP·HOG·Gabor·Minucje) i wynikami wszystkich 3 klasyfikatorów. "
        "Arkusz **Legenda** opisuje każdą kolumnę."
    )

    if st.button("💾 Zapisz ten wynik do Excela", type="primary", use_container_width=True):
        try:
            saved_path = write_detailed_prediction_excel(
                image_name=current_path.name,
                true_id=true_id,
                is_external=current_is_external,
                prep_result=result,
                raw_features=raw_feats,
                predictions=predictions,
                time_total_ms=time_total_ms,
            )
            st.success(f"✅ Zapisano do: `{saved_path}`")
            st.caption("Otwórz plik → arkusz **Predykcja** (dane) + **Legenda** (opisy kolumn).")
        except Exception as e:
            st.error(f"Błąd zapisu: {e}")
