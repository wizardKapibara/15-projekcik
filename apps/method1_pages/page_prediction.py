"""Metoda 1 — Strona: Predykcja (identyfikacja osoby po zdjęciu)."""

from __future__ import annotations

import random
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.core.dataset_loader import LipDataset
from src.core.filename_parser import try_parse_filename
from src.core.results_writer import ExperimentResults, PredictionRow
from src.method1_traditional.classifier import TraditionalClassifier
from src.method1_traditional.orb_compare import orb_keypoints_image
from apps.method1_pages._shared import get_dataset, get_metadata, meta_card


def render() -> None:
    st.title("Predykcja — Metoda 1 (Tradycyjna)")
    st.caption(
        "Zdjęcie porównywane jest ze wszystkimi zdjęciami w bazie za pomocą wybranej miary "
        "podobieństwa. Osoba o najwyższym score to wynik identyfikacji."
    )

    try:
        dataset = get_dataset()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    metadata = get_metadata()

    all_paths = dataset.all_paths()
    all_names = [p.name for p in all_paths]
    path_by_name = {p.name: p for p in all_paths}

    # ---- Wybór zdjęcia ----
    st.header("Wybór zdjęcia")
    tab1, tab2, tab3 = st.tabs(["🎲 Losowe z bazy", "📋 Wybór ręczny", "📤 Upload pliku"])

    selected_path: Path | None = None
    uploaded_bytes: bytes | None = None
    is_external = False

    with tab1:
        if "m1_rand" not in st.session_state:
            st.session_state.m1_rand = random.choice(all_names)
        if st.button("🎲 Wylosuj zdjęcie", use_container_width=True):
            st.session_state.m1_rand = random.choice(all_names)
        st.write(f"Wylosowane: `{st.session_state.m1_rand}`")
        if st.button("Użyj losowego →", type="primary", key="m1_use_rand"):
            selected_path = path_by_name[st.session_state.m1_rand]
            is_external = False
            uploaded_bytes = None

    with tab2:
        c1, c2 = st.columns([4, 1])
        man_name = c1.selectbox("Zdjęcie z bazy", all_names, key="m1_man_sel")
        c2.write(""); c2.write("")
        if c2.button("Użyj →", key="m1_use_man"):
            selected_path = path_by_name[man_name]
            is_external = False
            uploaded_bytes = None

    with tab3:
        up = st.file_uploader("Plik .png/.jpg spoza bazy", type=["png", "jpg", "jpeg"])
        if up is not None:
            if st.button("Użyj uploadowanego →", type="primary", key="m1_use_up"):
                selected_path = Path(up.name)
                is_external = True
                uploaded_bytes = up.getvalue()

    # Persist selection
    for key, default in [("m1_sel_path", None), ("m1_is_ext", False), ("m1_up_bytes", None)]:
        if key not in st.session_state:
            st.session_state[key] = default

    if selected_path is not None:
        st.session_state["m1_sel_path"] = selected_path
        st.session_state["m1_is_ext"] = is_external
        st.session_state["m1_up_bytes"] = uploaded_bytes

    cur_path: Path | None = st.session_state["m1_sel_path"]
    cur_ext: bool = st.session_state["m1_is_ext"]
    cur_up: bytes | None = st.session_state["m1_up_bytes"]

    if cur_path is None:
        st.info("Wybierz zdjęcie i kliknij 'Użyj →'.")
        return

    st.divider()

    # ---- Wczytaj obraz ----
    if cur_ext and cur_up:
        arr = np.frombuffer(cur_up, dtype=np.uint8)
        query_rgb = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        query_rgb = cv2.cvtColor(query_rgb, cv2.COLOR_BGR2RGB)
    else:
        query_rgb = LipDataset.load_image(cur_path, color="rgb")

    true_id: int | None = None
    if not cur_ext:
        parsed = try_parse_filename(cur_path)
        if parsed:
            true_id = parsed.person_id

    # ---- Ustawienia ----
    st.subheader("Ustawienia porównania")
    col_m, col_k = st.columns(2)
    method = col_m.radio(
        "Miara podobieństwa",
        ["ssim", "orb", "hist", "combined"],
        horizontal=True,
        format_func=lambda x: {"ssim": "SSIM", "orb": "ORB", "hist": "Histogram", "combined": "Combined"}[x],
    )
    top_k = col_k.slider("Liczba kandydatów (Top-K)", 1, 10, 5)

    if method == "orb":
        with st.expander("Wizualizacja punktów kluczowych ORB (query)"):
            kp_img, n_kp = orb_keypoints_image(query_rgb)
            st.image(kp_img, caption=f"Punkty kluczowe ORB: {n_kp}", use_container_width=True)

    st.write(f"**Zdjęcie:** `{cur_path.name}`")
    c_img, c_info = st.columns([2, 1])
    c_img.image(query_rgb, caption="Zdjęcie do identyfikacji", use_container_width=True)
    with c_info:
        if true_id:
            st.metric("Prawdziwa osoba", f"Nr_{true_id:02d}")
        st.metric("Metoda", method.upper())
        st.caption(
            "**SSIM** — strukturalne podobieństwo obrazów\n\n"
            "**ORB** — porównanie punktów kluczowych\n\n"
            "**Histogram** — podobieństwo rozkładu jasności\n\n"
            "**Combined** — ważona kombinacja wszystkich trzech"
        )

    if not st.button("🔍 Porównaj z bazą", type="primary", use_container_width=True):
        return

    # ---- Porównanie ----
    train_pairs = dataset.all_pairs()
    if not cur_ext:
        train_pairs = [(p, m) for p, m in train_pairs if str(p) != str(cur_path)]
    train_paths = [p for p, _ in train_pairs]
    train_labels = [m.person_id for _, m in train_pairs]

    clf = TraditionalClassifier()
    clf.fit(train_paths, train_labels)

    progress = st.progress(0.0, text="Porównuję ze zbiorem treningowym...")

    t0 = time.time()
    result = clf.predict(
        query_rgb, method=method, top_k=top_k,
        progress_callback=lambda i, n: progress.progress(i / n, text=f"Porównuję: {i}/{n}")
    )
    t1 = time.time()
    progress.progress(1.0, text="Gotowe!")

    predicted_id = result["predicted"]
    confidence = result["confidence"]
    top_k_results = result["top_k"]

    # ---- Wynik ----
    st.subheader("Wynik identyfikacji")
    col1, col2, col3 = st.columns(3)
    col1.metric("Predykcja (Top-1)", f"Nr_{predicted_id:02d}")
    col2.metric(f"Score ({method.upper()})", f"{confidence:.4f}")
    col3.metric("Czas", f"{(t1 - t0) * 1000:.0f} ms")

    if true_id is not None:
        if predicted_id == true_id:
            st.success(f"✅ POPRAWNIE rozpoznano jako Nr_{predicted_id:02d}")
        else:
            st.error(f"❌ BŁĄD: przewidywano Nr_{predicted_id:02d}, prawdziwa: Nr_{true_id:02d}")

    st.divider()

    # Top-K miniaturki
    st.subheader(f"Top-{top_k} kandydatów")
    cols = st.columns(min(top_k, 5))
    for i, (pid, score, img_idx) in enumerate(top_k_results[:5]):
        cols[i].image(clf._train_images[img_idx],
                      caption=f"Nr_{pid:02d}  {score:.4f}",
                      use_container_width=True)

    # Wykres scorów per osoba
    st.subheader("Wykres podobieństwa do każdej osoby")
    all_scores = result["all_scores"]
    person_best: dict[int, float] = {}
    for idx, lbl in enumerate(train_labels):
        sc = float(all_scores[idx])
        if lbl not in person_best or sc > person_best[lbl]:
            person_best[lbl] = sc

    persons_sorted = sorted(person_best.keys())
    bar_colors = [
        "#2E75B6" if p == predicted_id else ("#C6EFCE" if p == true_id else "#D9D9D9")
        for p in persons_sorted
    ]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([f"Nr_{p:02d}" for p in persons_sorted],
           [person_best[p] for p in persons_sorted], color=bar_colors)
    ax.set_xlabel("Osoba")
    ax.set_ylabel(f"Najlepszy score ({method.upper()})")
    ax.set_title(f"Podobieństwo do każdej osoby — {method.upper()}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
    st.caption("🔵 niebieski = predykcja top-1 | 🟢 zielony = prawdziwa osoba")

    # Metadane
    st.divider()
    col_pred_meta, col_true_meta = st.columns(2)
    with col_pred_meta:
        meta_card(predicted_id, metadata, "Przewidywana osoba")
    with col_true_meta:
        if true_id and true_id != predicted_id:
            meta_card(true_id, metadata, "Prawdziwa osoba")

    # Zapis
    st.divider()
    st.subheader("Zapis do Excela")
    if st.button("💾 Zapisz wynik do Excela", use_container_width=True):
        if "m1_experiment" not in st.session_state:
            st.session_state.m1_experiment = ExperimentResults("method1_experiment")
        exp: ExperimentResults = st.session_state.m1_experiment
        exp.add_prediction(PredictionRow(
            image_path=str(cur_path),
            true_id=true_id or -1,
            method=method,
            predicted_id=predicted_id,
            top3=[pid for pid, _, _ in top_k_results[:3]],
            top5=[pid for pid, _, _ in top_k_results[:5]],
            confidence=confidence,
            time_ms=(t1 - t0) * 1000,
            correct=(true_id == predicted_id) if true_id else False,
        ))
        saved = exp.save()
        st.success(f"Zapisano: `{saved}` | Predykcji: {len(exp.predictions)}")
