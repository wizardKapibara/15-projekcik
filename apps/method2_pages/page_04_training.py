"""Strona 4 — Trening klasyfikatora.

Funkcjonalności:
- Status datasetu (hash) — czy potrzebny retrain
- Wybór klasyfikatora (SVM / RF / k-NN / wszystkie 3)
- Trening na 1 foldzie (szybki podgląd) lub 5-fold CV (pełna ewaluacja)
- Pasek postępu + log per etap
- Zapis modeli do models/
- Metryki wynikowe: accuracy, per-fold variance
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.core.config import BAZA_DANYCH_DIR, MODELS_DIR
from src.core.dataset_hash import (
    compute_dataset_hash,
    is_retrain_needed,
    load_saved_hash,
    save_current_hash,
)
from src.core.dataset_loader import LipDataset
from src.core.splits import get_or_create_splits
from src.method2_ml.classifier import LipClassifier
from src.method2_ml.feature_vector import FeatureExtractorPipeline


@st.cache_resource
def _dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


def _log(container, msg: str) -> None:
    container.write(msg)


def _run_fold(
    fold,
    clf_name: str,
    pipeline: FeatureExtractorPipeline,
    log_container,
    progress_bar,
    step_start: float,
    step_end: float,
    force_recompute: bool = False,
) -> dict:
    """Trenuj i ewaluuj na jednym foldzie.

    Returns:
        Słownik z metrykami folda.
    """
    n_train = fold.n_train
    n_test = fold.n_test

    _log(log_container, f"  Ekstrakcja cech treningowych ({n_train} zdjęć)...")

    def _prog_train(i, n):
        frac = step_start + (step_end - step_start) * 0.6 * i / n
        progress_bar.progress(frac, text=f"Fold {fold.fold_index + 1}: ekstrakcja train {i}/{n}")

    t0 = time.time()
    pipeline.fit(fold.train_paths, force_recompute=force_recompute, progress_callback=_prog_train)
    X_train = pipeline.transform(fold.train_paths, force_recompute=force_recompute)
    y_train = np.array(fold.train_labels)

    _log(log_container, f"  Ekstrakcja cech testowych ({n_test} zdjęć)...")

    def _prog_test(i, n):
        frac = step_start + (step_end - step_start) * (0.6 + 0.2 * i / n)
        progress_bar.progress(frac, text=f"Fold {fold.fold_index + 1}: ekstrakcja test {i}/{n}")

    X_test = pipeline.transform(fold.test_paths, force_recompute=force_recompute, progress_callback=_prog_test)
    y_test = np.array(fold.test_labels)

    _log(log_container, f"  Trening {clf_name.upper()} ({X_train.shape})...")
    progress_bar.progress(
        step_start + (step_end - step_start) * 0.85,
        text=f"Fold {fold.fold_index + 1}: trening {clf_name.upper()}"
    )

    clf = LipClassifier(clf_name)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_top1(X_test)
    top3 = clf.top_k_accuracy(X_test, y_test, k=3)
    top5 = clf.top_k_accuracy(X_test, y_test, k=5)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    t1 = time.time()

    _log(log_container,
         f"  ✅ Fold {fold.fold_index + 1}: accuracy={acc:.3f} top3={top3:.3f} top5={top5:.3f} "
         f"F1={f1:.3f} ({t1 - t0:.1f}s)")

    progress_bar.progress(step_end, text=f"Fold {fold.fold_index + 1} ukończony")

    return {
        "fold": fold.fold_index + 1,
        "accuracy": acc,
        "top3": top3,
        "top5": top5,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "time_s": t1 - t0,
        "clf": clf,
        "X_test": X_test,
        "y_test": y_test,
    }


def render() -> None:
    st.title("Trening klasyfikatora")

    try:
        dataset = _dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ---- Status datasetu ----
    st.header("Status datasetu")
    current_hash = compute_dataset_hash()
    saved_hash = load_saved_hash()
    needs_retrain = is_retrain_needed()

    col1, col2 = st.columns(2)
    col1.metric("Zdjęć w bazie", dataset.total_count())
    col2.metric("Osób", len(dataset.list_persons()))

    if needs_retrain:
        st.warning(
            "⚠️ Wykryto zmiany w bazie danych lub model nie był jeszcze trenowany. "
            "Zalecany **retrain**."
        )
    else:
        st.success("✅ Model jest aktualny — baza danych nie zmieniła się od ostatniego trenowania.")

    with st.expander("Szczegóły hash"):
        st.caption(f"Hash aktualny:    `{current_hash}`")
        st.caption(f"Hash zapisany:    `{saved_hash or 'brak'}`")

    st.divider()

    # ---- Ustawienia treningu ----
    st.header("Ustawienia treningu")

    col_clf, col_mode = st.columns(2)
    with col_clf:
        clf_options = ["svm", "rf", "knn", "wszystkie 3"]
        clf_choice = st.radio(
            "Klasyfikator",
            clf_options,
            index=0,
            help="SVM: najlepsza dokładność | RF: odporny na szum | k-NN: prosty i interpretowalny"
        )

    with col_mode:
        train_mode = st.radio(
            "Tryb treningu",
            ["1 fold (szybki podgląd)", "5-fold CV (pełna ewaluacja)"],
            index=0,
            help="1 fold: ~2-5 min | 5-fold CV: ~10-25 min (zależy od sprzętu)"
        )
        single_fold_idx = None
        if "1 fold" in train_mode:
            single_fold_idx = st.number_input("Numer folda (0-4)", 0, 4, 0, 1)

    hog_pca = st.slider(
        "Liczba składowych PCA dla HOG",
        min_value=20, max_value=200, value=100, step=10,
        help="HOG daje ~16740 cech — PCA redukuje wymiarowość. Więcej składowych = więcej informacji, wolniejszy trening."
    )

    force_recompute = not st.checkbox(
        "Używaj cache wektorów cech",
        value=True,
        help="Jeśli True: wczytuje .npy z dysku zamiast przeliczać. Odznacz tylko jeśli zmieniłeś parametry preprocessingu."
    )

    # ---- Przycisk treningu ----
    st.divider()
    train_clfs = (
        ["svm", "rf", "knn"] if clf_choice == "wszystkie 3" else [clf_choice]
    )

    btn_label = f"🚀 Trenuj {clf_choice.upper()} ({'5-fold CV' if '5-fold' in train_mode else '1 fold'})"
    if not st.button(btn_label, type="primary", use_container_width=True):
        st.info("Kliknij powyżej aby rozpocząć trening.")
        return

    # ---- Trening ----
    folds = get_or_create_splits(dataset)
    folds_to_run = folds if "5-fold" in train_mode else [folds[single_fold_idx]]

    all_results: dict[str, list[dict]] = {c: [] for c in train_clfs}

    log_container = st.empty()
    st.write("---")
    progress_bar = st.progress(0.0, text="Inicjalizacja...")

    total_steps = len(train_clfs) * len(folds_to_run)
    step = 0
    last_pipeline: FeatureExtractorPipeline | None = None  # do zapisu po wszystkich klasyfikatorach

    for clf_name in train_clfs:
        st.write(f"### Klasyfikator: **{clf_name.upper()}**")

        for fold in folds_to_run:
            step_start = step / total_steps
            step_end = (step + 1) / total_steps

            _log(log_container, f"\n**Fold {fold.fold_index + 1}/{len(folds_to_run)}** — {clf_name.upper()}")

            pipeline = FeatureExtractorPipeline(hog_pca_components=hog_pca)
            res = _run_fold(fold, clf_name, pipeline, log_container, progress_bar,
                            step_start, step_end, force_recompute=force_recompute)
            all_results[clf_name].append(res)
            last_pipeline = pipeline  # zachowaj pipeline z ostatnio wytrenowanego folda
            step += 1

        # Zapis modelu (z ostatniego folda)
        last_clf = all_results[clf_name][-1]["clf"]
        saved_path = last_clf.save()
        _log(log_container, f"💾 Model {clf_name.upper()} zapisany: `{saved_path}`")

    # Zapisz pipeline (scaler + PCA) — BEZ ponownego fitowania, używamy już wytrenowanego
    if last_pipeline is not None:
        try:
            pipeline_path = last_pipeline.save()
            _log(log_container, f"💾 Pipeline (scaler+PCA) zapisany: `{pipeline_path}`")
        except Exception as e:
            _log(log_container, f"⚠️ Błąd zapisu pipeline: {e}")

    progress_bar.progress(1.0, text="Trening ukończony!")
    save_current_hash(current_hash)

    # ---- Wyniki ----
    st.success("✅ Trening zakończony!")
    st.header("Wyniki treningu")

    for clf_name in train_clfs:
        results = all_results[clf_name]
        st.subheader(f"{clf_name.upper()}")

        if len(results) == 1:
            r = results[0]
            st.write(f"Fold {r['fold']}: accuracy={r['accuracy']:.3f} top3={r['top3']:.3f} top5={r['top5']:.3f}")
        else:
            # Tabela per fold
            df = pd.DataFrame([{
                "Fold": r["fold"],
                "Accuracy": f"{r['accuracy']:.3f}",
                "Top-3": f"{r['top3']:.3f}",
                "Top-5": f"{r['top5']:.3f}",
                "F1 (macro)": f"{r['f1']:.3f}",
                "Czas [s]": f"{r['time_s']:.1f}",
            } for r in results])
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Agregat
            accs = [r["accuracy"] for r in results]
            top5s = [r["top5"] for r in results]
            f1s = [r["f1"] for r in results]
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy (mean ± std)",
                        f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
            col2.metric("Top-5 (mean ± std)",
                        f"{np.mean(top5s):.3f} ± {np.std(top5s):.3f}")
            col3.metric("F1 macro (mean ± std)",
                        f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    st.caption(
        f"Modele zapisane w: `{MODELS_DIR}` | "
        f"Hash datasetu zaktualizowany."
    )
