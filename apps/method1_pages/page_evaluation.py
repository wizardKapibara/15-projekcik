"""Metoda 1 — Strona: Ewaluacja (5-fold CV)."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import StratifiedKFold

from src.core.config import RANDOM_SEED
from src.core.results_writer import ExperimentResults, FoldSummary, PredictionRow
from src.method1_traditional.classifier import TraditionalClassifier
from apps.method1_pages._shared import get_dataset

_STATE_RESULTS = "eval_m1_results"
_STATE_EXPERIMENT = "eval_m1_experiment"
_STATE_METHODS = "eval_m1_methods"


def _run_evaluation(methods_to_eval: list[str], dataset) -> None:
    all_pairs = dataset.all_pairs()
    all_paths = [str(p) for p, _ in all_pairs]
    all_labels = [m.person_id for _, m in all_pairs]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(all_paths, all_labels))

    experiment = ExperimentResults("method1_cv")
    progress = st.progress(0.0)
    log = st.empty()

    total = len(methods_to_eval) * 5
    step = 0
    results_summary: dict[str, list[dict]] = {m: [] for m in methods_to_eval}

    for method in methods_to_eval:
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            progress.progress(step / total, text=f"{method.upper()} — Fold {fold_idx + 1}/5")
            log.write(f"**{method.upper()}** | Fold {fold_idx + 1}")

            train_paths_f = [all_paths[i] for i in train_idx]
            train_labels_f = [all_labels[i] for i in train_idx]
            test_paths_f  = [all_paths[i] for i in test_idx]
            test_labels_f  = [all_labels[i] for i in test_idx]

            clf = TraditionalClassifier()
            clf.fit(train_paths_f, train_labels_f)

            t0 = time.time()
            fold_result = clf.score(test_paths_f, test_labels_f, method=method)
            t1 = time.time()

            acc  = fold_result["accuracy"]
            top3 = fold_result["top3"]
            top5 = fold_result["top5"]

            results_summary[method].append({
                "fold": fold_idx + 1, "acc": acc, "top3": top3, "top5": top5,
                "time_s": t1 - t0,
            })
            experiment.add_fold_summary(FoldSummary(
                fold=fold_idx + 1, method=method,
                accuracy=acc, top3_accuracy=top3, top5_accuracy=top5,
                precision=0.0, recall=0.0, f1=0.0, time_s=t1 - t0,
            ))
            for pred in fold_result["predictions"]:
                experiment.add_prediction(PredictionRow(
                    image_path="", true_id=pred["true_id"], method=method,
                    predicted_id=pred["predicted"],
                    top3=pred["top5"][:3], top5=pred["top5"],
                    confidence=pred["confidence"],
                    time_ms=0.0, correct=pred["correct"],
                ))
            step += 1

    progress.progress(1.0, text="Gotowe!")
    log.empty()

    st.session_state[_STATE_RESULTS] = results_summary
    st.session_state[_STATE_EXPERIMENT] = experiment
    st.session_state[_STATE_METHODS] = methods_to_eval


def _show_results(results_summary: dict, methods_to_eval: list[str]) -> None:
    st.success("✅ Ewaluacja zakończona!")
    st.header("Wyniki porównania miar")

    comparison = []
    for method in methods_to_eval:
        fold_data = results_summary[method]
        accs  = [r["acc"]  for r in fold_data]
        top3s = [r["top3"] for r in fold_data]
        top5s = [r["top5"] for r in fold_data]
        comparison.append({
            "Metoda":                  method.upper(),
            "Accuracy (mean ± std)":   f"{np.mean(accs):.3f} ± {np.std(accs):.3f}",
            "Top-3 (mean)":            f"{np.mean(top3s):.3f}",
            "Top-5 (mean)":            f"{np.mean(top5s):.3f}",
            "Czas/fold [s]":           f"{np.mean([r['time_s'] for r in fold_data]):.1f}",
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    for method in methods_to_eval:
        with st.expander(f"Szczegóły per fold — {method.upper()}"):
            rows = results_summary[method]
            df = pd.DataFrame([{
                "Fold":     r["fold"],
                "Accuracy": f"{r['acc']:.3f}",
                "Top-3":    f"{r['top3']:.3f}",
                "Top-5":    f"{r['top5']:.3f}",
                "Czas [s]": f"{r['time_s']:.1f}",
            } for r in rows])
            st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    if st.button("📊 Eksportuj wyniki do Excela", use_container_width=True):
        saved = st.session_state[_STATE_EXPERIMENT].save()
        st.success(f"Zapisano: `{saved}`")


def render() -> None:
    st.title("Ewaluacja — Metoda 1 (5-fold CV)")
    st.caption(
        "Porównanie czterech miar podobieństwa (SSIM, ORB, Histogram, Combined) "
        "na tym samym podziale 5-fold CV. "
        "Każde zdjęcie testowe porównywane jest ze wszystkimi treningowymi — może potrwać kilkanaście minut."
    )

    try:
        dataset = get_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    st.info(
        "**Jak działa Metoda 1?**  \n"
        "Dla każdego zdjęcia testowego obliczany jest score podobieństwa do **każdego** zdjęcia "
        "treningowego. Osoba z najwyższym score wygrywa (zasada 1-NN)."
    )

    methods_to_eval = st.multiselect(
        "Miary do ewaluacji",
        ["ssim", "orb", "hist", "combined"],
        default=["ssim", "hist"],
        help="ORB i Combined są wolniejsze. Zacznij od SSIM + Histogram.",
    )
    if not methods_to_eval:
        st.warning("Wybierz co najmniej jedną miarę.")
        return

    if st.button("🔬 Uruchom ewaluację 5-fold CV", type="primary", use_container_width=True):
        st.session_state.pop(_STATE_RESULTS, None)
        st.session_state.pop(_STATE_EXPERIMENT, None)
        st.session_state.pop(_STATE_METHODS, None)
        with st.spinner("Trwa ewaluacja..."):
            _run_evaluation(methods_to_eval, dataset)

    if _STATE_RESULTS in st.session_state:
        _show_results(
            st.session_state[_STATE_RESULTS],
            st.session_state[_STATE_METHODS],
        )
    else:
        st.info("Kliknij przycisk powyżej, aby uruchomić pełną ewaluację. Czas: ~2–5 min dla SSIM/Histogram.")
