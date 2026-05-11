"""Pełna ewaluacja obu metod — bez Streamlit.

Uruchomienie:
    python evaluate_all.py

Wyniki zapisywane do:
    results/experiment_<TIMESTAMP>.xlsx
    results/plots/confusion_matrix_*.png
"""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # bez GUI
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from src.core.config import BAZA_DANYCH_DIR, PLOTS_DIR, RANDOM_SEED
from src.core.dataset_loader import LipDataset
from src.core.results_writer import ExperimentResults, FoldSummary, PredictionRow
from src.core.splits import get_or_create_splits
from src.method1_traditional.classifier import TraditionalClassifier
from src.method2_ml.classifier import LipClassifier
from src.method2_ml.feature_vector import FeatureExtractorPipeline

HOG_PCA = 100
M1_METHODS = ["ssim", "orb", "hist", "combined"]
M2_CLFS = ["svm", "rf", "knn"]


def _print_header(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def _save_confusion_matrix(cm: np.ndarray, classes: list, clf_name: str) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [f"Nr_{int(c):02d}" for c in classes]
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.3
    )
    ax.set_xlabel("Przewidywana klasa")
    ax.set_ylabel("Prawdziwa klasa")
    ax.set_title(f"Confusion Matrix — {clf_name.upper()} (5-fold CV)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = PLOTS_DIR / f"confusion_matrix_{clf_name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Wykres: {path}")


def evaluate_method2(dataset: LipDataset, folds, experiment: ExperimentResults) -> dict:
    """5-fold CV dla SVM / RF / k-NN."""
    _print_header("METODA 2 — Klasyfikacja ML (5-fold CV)")
    summary = {}

    for clf_name in M2_CLFS:
        print(f"\n[{clf_name.upper()}]")
        fold_accs, fold_top3, fold_top5, fold_f1 = [], [], [], []
        cm_total = None
        classes_global = None

        for fold in folds:
            t0 = time.time()
            print(f"  Fold {fold.fold_index + 1}/5 — ekstrakcja cech...", end=" ", flush=True)

            pipeline = FeatureExtractorPipeline(hog_pca_components=HOG_PCA)
            pipeline.fit(fold.train_paths, force_recompute=False)
            X_train = pipeline.transform(fold.train_paths, force_recompute=False)
            X_test = pipeline.transform(fold.test_paths, force_recompute=False)
            y_train = np.array(fold.train_labels)
            y_test = np.array(fold.test_labels)

            print("trening...", end=" ", flush=True)
            clf = LipClassifier(clf_name)
            clf.fit(X_train, y_train)

            y_pred = clf.predict_top1(X_test)
            y_proba = clf.predict_proba(X_test)
            classes = clf._clf.classes_

            acc = accuracy_score(y_test, y_pred)
            top3 = clf.top_k_accuracy(X_test, y_test, k=3)
            top5 = clf.top_k_accuracy(X_test, y_test, k=5)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
            t1 = time.time()

            cm = confusion_matrix(y_test, y_pred, labels=classes)
            cm_total = cm if cm_total is None else cm_total + cm
            classes_global = classes

            fold_accs.append(acc)
            fold_top3.append(top3)
            fold_top5.append(top5)
            fold_f1.append(f1)

            print(f"acc={acc:.3f}  top3={top3:.3f}  top5={top5:.3f}  F1={f1:.3f}  ({t1-t0:.0f}s)")

            experiment.add_fold_summary(FoldSummary(
                fold=fold.fold_index + 1, method=clf_name,
                accuracy=acc, top3_accuracy=top3, top5_accuracy=top5,
                precision=prec, recall=rec, f1=f1, time_s=t1 - t0,
            ))

            sorted_idx = np.argsort(y_proba, axis=1)[:, ::-1]
            for i, (path, true_lbl, pred_lbl) in enumerate(
                zip(fold.test_paths, fold.test_labels, y_pred.tolist())
            ):
                pred_class_idx = int(np.where(classes == pred_lbl)[0][0])
                experiment.add_prediction(PredictionRow(
                    image_path=path, true_id=int(true_lbl), method=clf_name,
                    predicted_id=int(pred_lbl),
                    top3=[int(c) for c in classes[sorted_idx[i, :3]]],
                    top5=[int(c) for c in classes[sorted_idx[i, :5]]],
                    confidence=float(y_proba[i, pred_class_idx]), time_ms=0.0,
                    correct=(int(true_lbl) == int(pred_lbl)),
                ))

        experiment.add_confusion_matrix(clf_name, cm_total, classes_global.tolist())
        _save_confusion_matrix(cm_total, classes_global.tolist(), clf_name)

        summary[clf_name] = {
            "acc": np.mean(fold_accs), "acc_std": np.std(fold_accs),
            "top3": np.mean(fold_top3), "top5": np.mean(fold_top5),
            "f1": np.mean(fold_f1),
        }
        print(f"  MEAN: acc={np.mean(fold_accs):.3f}±{np.std(fold_accs):.3f}  "
              f"top3={np.mean(fold_top3):.3f}  top5={np.mean(fold_top5):.3f}  "
              f"F1={np.mean(fold_f1):.3f}")

    return summary


def evaluate_method1(dataset: LipDataset, folds, experiment: ExperimentResults) -> dict:
    """5-fold CV dla SSIM / ORB / Hist / Combined."""
    _print_header("METODA 1 — Porównanie tradycyjne (5-fold CV)")
    summary = {}

    for method in M1_METHODS:
        print(f"\n[{method.upper()}]")
        fold_accs, fold_top3, fold_top5 = [], [], []

        for fold in folds:
            t0 = time.time()
            print(f"  Fold {fold.fold_index + 1}/5...", end=" ", flush=True)

            clf = TraditionalClassifier()
            clf.fit(fold.train_paths, fold.train_labels)
            result = clf.score(fold.test_paths, fold.test_labels, method=method)

            acc = result["accuracy"]
            top3 = result["top3"]
            top5 = result["top5"]
            t1 = time.time()

            fold_accs.append(acc)
            fold_top3.append(top3)
            fold_top5.append(top5)

            print(f"acc={acc:.3f}  top3={top3:.3f}  top5={top5:.3f}  ({t1-t0:.0f}s)")

            experiment.add_fold_summary(FoldSummary(
                fold=fold.fold_index + 1, method=f"m1_{method}",
                accuracy=acc, top3_accuracy=top3, top5_accuracy=top5,
                precision=0.0, recall=0.0, f1=0.0, time_s=t1 - t0,
            ))

            for pred in result["predictions"]:
                experiment.add_prediction(PredictionRow(
                    image_path="", true_id=pred["true_id"], method=f"m1_{method}",
                    predicted_id=pred["predicted"],
                    top3=pred["top5"][:3], top5=pred["top5"],
                    confidence=float(pred["confidence"]), time_ms=0.0,
                    correct=pred["correct"],
                ))

        summary[method] = {
            "acc": np.mean(fold_accs), "top3": np.mean(fold_top3), "top5": np.mean(fold_top5),
        }
        print(f"  MEAN: acc={np.mean(fold_accs):.3f}  top3={np.mean(fold_top3):.3f}  "
              f"top5={np.mean(fold_top5):.3f}")

    return summary


def print_final_table(m1: dict, m2: dict) -> None:
    _print_header("PODSUMOWANIE WSTĘPNYCH BADAŃ")

    print("\nMetoda 1 — Porównanie tradycyjne (1-NN):")
    print(f"  {'Metoda':<12} {'Accuracy':>10} {'Top-3':>8} {'Top-5':>8}")
    print(f"  {'-'*40}")
    for method, r in m1.items():
        print(f"  {method.upper():<12} {r['acc']*100:>9.1f}% {r['top3']*100:>7.1f}% {r['top5']*100:>7.1f}%")

    print("\nMetoda 2 — Klasyfikacja ML (5-fold CV, mean ± std):")
    print(f"  {'Klasyfikator':<12} {'Accuracy':>16} {'Top-3':>8} {'Top-5':>8} {'F1':>8}")
    print(f"  {'-'*56}")
    for clf, r in m2.items():
        acc_str = f"{r['acc']*100:.1f}±{r['acc_std']*100:.1f}%"
        print(f"  {clf.upper():<12} {acc_str:>16} {r['top3']*100:>7.1f}% "
              f"{r['top5']*100:>7.1f}% {r['f1']*100:>7.1f}%")

    print(f"\n  Baseline losowy: {100/22:.1f}%  (1/22 klas)")
    print()


def main() -> None:
    print("Wstępne badania — identyfikacja na podstawie fotografii ust")
    print(f"HOG PCA: {HOG_PCA} komponentów | Seed: {RANDOM_SEED}")

    dataset = LipDataset(BAZA_DANYCH_DIR)
    folds = get_or_create_splits(dataset)
    print(f"Dataset: {len(folds[0].train_paths) + len(folds[0].test_paths)} obrazów, "
          f"5-fold CV ({len(folds[0].train_paths)} train / {len(folds[0].test_paths)} test)")

    experiment = ExperimentResults()
    experiment.add_run_info("HOG_PCA_components", str(HOG_PCA))
    experiment.add_run_info("Klasyfikatory_M2", ", ".join(M2_CLFS))
    experiment.add_run_info("Metody_M1", ", ".join(M1_METHODS))
    experiment.add_run_info("N_folds", "5")

    m2_summary = evaluate_method2(dataset, folds, experiment)

    # Metoda 1 jest wolna (porównanie każdego z każdym) — można pominąć
    run_m1 = input("\nUruchomić też Metodę 1 (SSIM/ORB)? Wolna — ~20-60 min. [t/N]: ").strip().lower()
    m1_summary = {}
    if run_m1 == "t":
        m1_summary = evaluate_method1(dataset, folds, experiment)
    else:
        print("Pomijam Metodę 1.")

    saved = experiment.save()
    print(f"\nWyniki zapisane: {saved}")

    print_final_table(m1_summary, m2_summary)


if __name__ == "__main__":
    main()
