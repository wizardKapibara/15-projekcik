"""Wspólne zasoby dla stron Metody 1 (cache datasetu, metadanych, meta_card)."""

from __future__ import annotations

import streamlit as st

from src.core.config import BAZA_DANYCH_DIR, METADATA_XLSM
from src.core.dataset_loader import LipDataset
from src.core.metadata_loader import load_person_metadata


@st.cache_resource
def get_dataset() -> LipDataset:
    return LipDataset(BAZA_DANYCH_DIR)


@st.cache_resource
def get_metadata() -> dict:
    if not METADATA_XLSM.exists():
        return {}
    return load_person_metadata(METADATA_XLSM)


def meta_card(person_id: int, metadata: dict, label: str) -> None:
    st.write(f"**{label}: Nr_{person_id:02d}**")
    meta = metadata.get(person_id)
    if meta is None:
        st.caption("Brak metadanych.")
        return
    for k, v in meta.to_dict().items():
        if k == "person_id" or v is None:
            continue
        st.caption(f"**{k}**: {v}")
