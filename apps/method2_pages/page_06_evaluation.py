"""Strona 6 — Ewaluacja i wyniki badań.

5-fold CV, accuracy, precision/recall/F1, macierz pomyłek,
top-3/top-5 accuracy, eksport do Excela i wykresów PNG.
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.core.config import BAZA_DANYCH_DIR, MODELS_DIR, PLOTS_DIR, RESULTS_DIR
from src.core.dataset_loader import LipDataset
from src.core.results_writer import ExperimentResults, FoldSummary, PredictionRow
from src.core.splits import get_or_create_splits
from src.method2_ml.classifier import AVAILABLE_CLASSIFIERS, LipClassifier
from src.method2_ml.feature_vector import FeatureExtractorPipeline

_STATE_RESULTS = "eval_m2_results"
_STATE_EXPERIMENT = "eval_m2_experiment"
_STATE_CLF_NAMES = "eval_m2_clf_names"


@st.cache_resource
def _dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


def _run_evaluation(clf_names: list[str], hog_pca: int, use_cache: bool) -> None:
    """Uruchamia ewaluację i zapisuje wyniki w st.session_state."""
    dataset = _dataset()
    folds = get_or_create_splits(dataset)
    experiment = ExperimentResults()
    experiment.add_run_info("HOG_PCA_components", str(hog_pca))
    experiment.add_run_info("Klasyfikatory", ", ".join(clf_names))
    experiment.add_run_info("N_folds", "5")

    all_results: dict = {}

    progress = st.progress(0.0, text="Inicjalizacja...")
    log = st.empty()

    total_steps = len(clf_names) * len(folds)
    step = 0

    for clf_name in clf_names:
        fold_results = []

        for fold in folds:
            frac_start = step / total_steps
            frac_end = (step + 1) / total_steps

            progress.progress(
                frac_start,
                text=f"{clf_name.upper()} — Fold {fold.fold_index + 1}/5: ekstrakcja cech...",
            )
            log.write(f"**{clf_name.upper()}** | Fold {fold.fold_index + 1}")

            pipeline = FeatureExtractorPipeline(hog_pca_components=hog_pca)
            t0 = time.time()
            pipeline.fit(fold.train_paths, force_recompute=not use_cache)
            X_train = pipeline.transform(fold.train_paths, force_recompute=not use_cache)
            y_train = np.array(fold.train_labels)

            progress.progress(
                frac_start + (frac_end - frac_start) * 0.7,
                text=f"{clf_name.upper()} — Fold {fold.fold_index + 1}/5: trening...",
            )

            X_test = pipeline.transform(fold.test_paths, force_recompute=not use_cache)
            y_test = np.array(fold.test_labels)

            fold_clf = LipClassifier(clf_name)
            fold_clf.fit(X_train, y_train)

            y_pred = fold_clf.predict_top1(X_test)
            y_proba = fold_clf.predict_proba(X_test)
            classes = fold_clf._clf.classes_

            acc = accuracy_score(y_test, y_pred)
            top3 = fold_clf.top_k_accuracy(X_test, y_test, k=3)
            top5 = fold_clf.top_k_accuracy(X_test, y_test, k=5)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            t1 = time.time()
            cm = confusion_matrix(y_test, y_pred, labels=classes)

            fold_results.append({
                "fold": fold.fold_index + 1,
                "acc": acc, "top3": top3, "top5": top5,
                "prec": prec, "rec": rec, "f1": f1,
                "time_s": t1 - t0,
                "cm": cm, "classes": classes,
                "y_test": y_test, "y_pred": y_pred,
            })

            experiment.add_fold_summary(FoldSummary(
                fold=fold.fold_index + 1, method=clf_name,
                accuracy=acc, top3_accuracy=top3, top5_accuracy=top5,
                precision=prec, recall=rec, f1=f1, time_s=t1 - t0,
            ))

            sorted_proba_idx = np.argsort(y_proba, axis=1)[:, ::-1]
            for i, (path, true_lbl, pred_lbl) in enumerate(
                zip(fold.test_paths, fold.test_labels, y_pred.tolist())
            ):
                pred_class_idx = np.where(classes == pred_lbl)[0][0]
                experiment.add_prediction(PredictionRow(
                    image_path=path, true_id=int(true_lbl), method=clf_name,
                    predicted_id=int(pred_lbl),
                    top3=[int(c) for c in classes[sorted_proba_idx[i, :3]]],
                    top5=[int(c) for c in classes[sorted_proba_idx[i, :5]]],
                    confidence=float(y_proba[i, pred_class_idx]), time_ms=0.0,
                    correct=(int(true_lbl) == int(pred_lbl)),
                ))

            step += 1
            progress.progress(frac_end)

        all_results[clf_name] = fold_results

        cm_total = sum(r["cm"] for r in fold_results)
        experiment.add_confusion_matrix(clf_name, cm_total, fold_results[0]["classes"].tolist())

    progress.progress(1.0, text="Ewaluacja zakończona!")
    log.empty()

    st.session_state[_STATE_RESULTS] = all_results
    st.session_state[_STATE_EXPERIMENT] = experiment
    st.session_state[_STATE_CLF_NAMES] = clf_names


def _show_results(all_results: dict, clf_names: list[str]) -> None:
    """Wyświetla wyniki przechowywane w session_state."""
    st.success("✅ Ewaluacja zakończona!")
    st.header("Wyniki ewaluacji")

    for clf_name in clf_names:
        results = all_results[clf_name]
        st.subheader(f"Klasyfikator: {clf_name.upper()}")

        df = pd.DataFrame([{
            "Fold": r["fold"],
            "Accuracy": f"{r['acc']:.3f}",
            "Top-3": f"{r['top3']:.3f}",
            "Top-5": f"{r['top5']:.3f}",
            "Precision": f"{r['prec']:.3f}",
            "Recall": f"{r['rec']:.3f}",
            "F1 (macro)": f"{r['f1']:.3f}",
            "Czas [s]": f"{r['time_s']:.1f}",
        } for r in results])
        st.dataframe(df, use_container_width=True, hide_index=True)

        accs = [r["acc"] for r in results]
        top5s = [r["top5"] for r in results]
        f1s = [r["f1"] for r in results]
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy (mean ± std)", f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
        col2.metric("Top-5 (mean ± std)", f"{np.mean(top5s):.3f} ± {np.std(top5s):.3f}")
        col3.metric("F1 macro (mean ± std)", f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

        st.write("**Macierz pomyłek (suma 5 foldów)**")
        cm_total = sum(r["cm"] for r in results)
        classes_labels = [f"Nr_{c:02d}" for c in results[0]["classes"]]

        fig, ax = plt.subplots(figsize=(14, 11))
        sns.heatmap(
            cm_total, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes_labels, yticklabels=classes_labels,
            ax=ax, linewidths=0.3,
        )
        ax.set_xlabel("Przewidywana klasa")
        ax.set_ylabel("Prawdziwa klasa")
        ax.set_title(f"Confusion Matrix — {clf_name.upper()} (5-fold CV)")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig)

        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_path = PLOTS_DIR / f"confusion_matrix_{clf_name}.png"
        fig.savefig(str(plot_path), dpi=150, bbox_inches="tight")
        st.caption(f"Wykres zapisany: `{plot_path}`")
        plt.close(fig)

        st.divider()

    if len(clf_names) > 1:
        st.header("Porównanie klasyfikatorów")
        comparison = []
        for clf_name in clf_names:
            results = all_results[clf_name]
            accs = [r["acc"] for r in results]
            top5s = [r["top5"] for r in results]
            f1s = [r["f1"] for r in results]
            comparison.append({
                "Klasyfikator": clf_name.upper(),
                "Accuracy (mean)": f"{np.mean(accs):.3f}",
                "Accuracy (std)": f"{np.std(accs):.3f}",
                "Top-5 (mean)": f"{np.mean(top5s):.3f}",
                "F1 macro (mean)": f"{np.mean(f1s):.3f}",
            })
        st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("Ewaluacja i wyniki badań")
    st.caption(
        "Pełna ewaluacja klasyfikatora — 5-fold CV, macierze pomyłek, "
        "eksport wyników do Excela i wykresów PNG do pracy magisterskiej."
    )

    try:
        dataset = _dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ---- Konfiguracja ----
    st.header("Konfiguracja ewaluacji")

    col1, col2 = st.columns(2)
    with col1:
        clf_options = AVAILABLE_CLASSIFIERS + ["wszystkie 3"]
        clf_choice = st.radio("Klasyfikator do ewaluacji", clf_options, index=0)
    with col2:
        hog_pca = st.slider(
            "HOG PCA komponentów", 20, 200, 100, 10,
            help="Musi być zgodne z parametrami z treningu!",
        )
        use_cache = st.checkbox("Używaj cache wektorów cech", value=True)

    clf_names = AVAILABLE_CLASSIFIERS if clf_choice == "wszystkie 3" else [clf_choice]

    missing = [c for c in clf_names if not LipClassifier.is_saved(c)]
    if missing:
        st.error(
            f"Brak wytrenowanych modeli: {', '.join(m.upper() for m in missing)}. "
            "Najpierw przejdź do strony **Trening modelu**."
        )
        return

    # ---- Przycisk uruchomienia ----
    st.divider()
    if st.button("🔬 Uruchom ewaluację 5-fold CV", type="primary", use_container_width=True):
        # Wyczyść poprzednie wyniki przed nowym przebiegiem
        st.session_state.pop(_STATE_RESULTS, None)
        st.session_state.pop(_STATE_EXPERIMENT, None)
        st.session_state.pop(_STATE_CLF_NAMES, None)
        with st.spinner("Trwa ewaluacja..."):
            _run_evaluation(clf_names, hog_pca, use_cache)

    # ---- Wyświetl wyniki (też po re-runie Streamlit) ----
    if _STATE_RESULTS in st.session_state:
        _show_results(
            st.session_state[_STATE_RESULTS],
            st.session_state[_STATE_CLF_NAMES],
        )

        # ---- Eksport ----
        st.divider()
        st.header("Eksport wyników")
        if st.button("📊 Eksportuj pełny eksperyment do Excela", use_container_width=True):
            saved = st.session_state[_STATE_EXPERIMENT].save()
            st.success(f"Zapisano: `{saved}`")
            experiment = st.session_state[_STATE_EXPERIMENT]
            st.caption(
                f"Arkusze: Run_info, Predictions ({len(experiment.predictions)} wierszy), "
                "Summary per metoda, macierze pomyłek."
            )
    else:
        st.info("Kliknij przycisk powyżej, aby uruchomić pełną ewaluację.")
