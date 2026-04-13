"""Strona 1 — Eksploracja danych.

Statystyki bazy + galeria zdjęć z metadanymi z Excela.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

from src.core.config import BAZA_DANYCH_DIR, METADATA_XLSM
from src.core.dataset_loader import LipDataset
from src.core.metadata_loader import PersonMetadata, load_dictionaries, load_person_metadata


@st.cache_resource
def _load_dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


@st.cache_resource
def _load_metadata() -> Dict[int, PersonMetadata]:
    if not METADATA_XLSM.exists():
        return {}
    return load_person_metadata(METADATA_XLSM)


@st.cache_resource
def _load_dicts():
    if not METADATA_XLSM.exists():
        return None
    return load_dictionaries(METADATA_XLSM)


def render() -> None:
    st.title("Eksploracja danych")
    st.caption("Przegląd bazy zdjęć ust i metadanych osób")

    try:
        dataset = _load_dataset()
    except FileNotFoundError as e:
        st.error(f"Nie znaleziono bazy danych: {e}")
        st.info(f"Spodziewana ścieżka: {BAZA_DANYCH_DIR}")
        return

    metadata = _load_metadata()
    dicts = _load_dicts()

    persons = dataset.list_persons()
    counts = dataset.count_per_person()

    # ----- Statystyki ogólne -----
    st.header("Statystyki ogólne")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Liczba osób", len(persons))
    col2.metric("Łącznie zdjęć", dataset.total_count())
    col3.metric("Min/osobę", min(counts.values()) if counts else 0)
    col4.metric("Max/osobę", max(counts.values()) if counts else 0)

    avg = dataset.total_count() / len(persons) if persons else 0
    st.write(f"Średnia liczba zdjęć na osobę: **{avg:.1f}**")

    # ----- Zdjęcia per osoba -----
    st.header("Liczba zdjęć per osoba")
    counts_df = pd.DataFrame(
        {"osoba": [f"Nr_{p:02d}" for p in persons], "liczba_zdjec": [counts[p] for p in persons]}
    )
    st.bar_chart(counts_df.set_index("osoba"), height=300)

    # ----- Rozkłady warunków akwizycji -----
    st.header("Warunki akwizycji")
    all_meta = dataset.all_pairs()

    flash_counter = Counter(p.flash for _, p in all_meta)
    focus_counter = Counter(round(p.focus, 2) for _, p in all_meta)
    exposure_counter = Counter(p.exposure for _, p in all_meta)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Lampa błyskowa")
        flash_df = pd.DataFrame(
            {"flash": list(flash_counter.keys()), "liczba": list(flash_counter.values())}
        )
        st.bar_chart(flash_df.set_index("flash"))
    with c2:
        st.subheader("Wartości fokusu")
        focus_df = pd.DataFrame(
            {"fokus": [str(f) for f in sorted(focus_counter.keys())],
             "liczba": [focus_counter[f] for f in sorted(focus_counter.keys())]}
        )
        st.bar_chart(focus_df.set_index("fokus"))
    with c3:
        st.subheader("Ekspozycja")
        exp_df = pd.DataFrame(
            {"exposure": [str(e) for e in sorted(exposure_counter.keys())],
             "liczba": [exposure_counter[e] for e in sorted(exposure_counter.keys())]}
        )
        st.bar_chart(exp_df.set_index("exposure"))

    st.divider()

    # ----- Galeria osoby -----
    st.header("Galeria osoby")
    person_options = {f"Nr_{p:02d}": p for p in persons}
    selected_label = st.selectbox(
        "Wybierz osobę", list(person_options.keys()), index=0
    )
    selected_id = person_options[selected_label]

    # Metadane z Excela
    st.subheader("Metadane (z pliku Excel)")
    person_meta = metadata.get(selected_id)
    if person_meta is None:
        st.warning("Brak metadanych w pliku Excel dla tej osoby.")
    else:
        meta_dict = person_meta.to_dict()
        meta_df = pd.DataFrame(
            [{"pole": k, "wartość": v} for k, v in meta_dict.items() if k != "person_id"]
        )
        st.dataframe(meta_df, use_container_width=True, hide_index=True)

        # Tłumaczenie unique_characteristics na PL jeśli mamy słownik
        if dicts and person_meta.unique_characteristics:
            pl = dicts.unique_characteristics.get(person_meta.unique_characteristics)
            if pl and pl != person_meta.unique_characteristics:
                st.caption(f"Cecha unikalna (PL): **{pl}**")

    # Zdjęcia osoby
    st.subheader(f"Zdjęcia ({counts[selected_id]} szt.)")
    image_paths = dataset.list_images(selected_id)
    image_meta = dataset.list_metadata(selected_id)

    n_cols = 4
    for i in range(0, len(image_paths), n_cols):
        cols = st.columns(n_cols)
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(image_paths):
                break
            path = image_paths[idx]
            meta = image_meta[idx]
            caption = (
                f"#{idx + 1} | Lampa {meta.flash} | "
                f"Fokus {meta.focus} | Exp {meta.exposure}"
            )
            try:
                img = LipDataset.load_image(path, color="rgb")
                col.image(img, caption=caption, use_container_width=True)
            except Exception as e:  # noqa: BLE001
                col.error(f"Błąd wczytywania: {e}")
