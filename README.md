# System cheiloskopijnej identyfikacji osób

## Praca magisterska — porównanie metod biometrii ust

Projekt implementuje i porównuje **dwie metody automatycznej identyfikacji osoby na podstawie zdjęcia ust (cheiloskopia / lip biometrics)**:

- **Metoda 1 — tradycyjne porównanie obrazów**: SSIM, ORB, Histogram — klasyfikacja przez 1-NN bez żadnej ekstrakcji cech; bezpośrednie porównanie pikseli i struktur.
- **Metoda 2 — klasyczne Machine Learning na ręcznie wyciągniętych cechach**: LBP + HOG + Gabor + minucje → SVM / Random Forest / k-NN.

Cel naukowy: identyfikacja **1-do-N** (zdjęcie ust → osoba z 22) na zbiorze 382 zdjęć PNG od 22 osób (~17 zdjęć/osobę).

> **Kluczowa decyzja naukowa**: metadane z arkusza Excel (płeć, wiek, odcień skóry, cechy szczególne) **NIE wchodzą do wektora cech** — byłby to data leakage psujący wartość badania. Służą wyłącznie jako kontekst w GUI i do analizy post-hoc.

---

## Spis treści

1. [Tło naukowe — czym jest cheiloskopia](#1-tło-naukowe--czym-jest-cheiloskopia)
2. [Wymagania i instalacja](#2-wymagania-i-instalacja)
3. [Uruchamianie aplikacji](#3-uruchamianie-aplikacji)
4. [Struktura projektu](#4-struktura-projektu)
5. [Dane wejściowe](#5-dane-wejściowe)
6. [Moduły wspólne — `src/core/`](#6-moduły-wspólne--srccore)
7. [Metoda 1 — tradycyjne porównanie obrazów](#7-metoda-1--tradycyjne-porównanie-obrazów)
8. [Metoda 2 — ML na cechach biometrycznych](#8-metoda-2--ml-na-cechach-biometrycznych)
9. [Schemat przepływu danych](#9-schemat-przepływu-danych)
10. [Aplikacja Streamlit — opis każdej strony](#10-aplikacja-streamlit--opis-każdej-strony)
11. [Pliki wyników i format Excela](#11-pliki-wyników-i-format-excela)
12. [Konwencje nazewnictwa zdjęć](#12-konwencje-nazewnictwa-zdjęć)
13. [Mechanizm auto-retrain (hash datasetu)](#13-mechanizm-auto-retrain-hash-datasetu)
14. [Typowe czasy działania](#14-typowe-czasy-działania)
15. [Konfiguracja projektu](#15-konfiguracja-projektu)
16. [Biblioteki i technologie — szczegółowy opis](#16-biblioteki-i-technologie--szczegółowy-opis)
17. [Ograniczenia projektu](#17-ograniczenia-projektu)
18. [Rozwiązywanie problemów](#18-rozwiązywanie-problemów)
19. [Różnice względem pierwotnych notatek projektowych](#19-różnice-względem-pierwotnych-notatek-projektowych)

---

## 1. Tło naukowe — czym jest cheiloskopia

**Cheiloskopia** (gr. *cheilos* = usta, *skopein* = badać) to dział kryminalistyki zajmujący się identyfikacją osób na podstawie wzorów bruzd ust (linii na wargach). Analogicznie do daktyloskopii (odcisków palców) oraz irydologii (tęczówki oka), wzory bruzd ust są:

- **unikalne dla każdej osoby** — nie ma dwóch ludzi o identycznym wzorcu,
- **niezmienne przez całe życie** — poza poważnymi patologiami lub urazami,
- **możliwe do odtworzenia** z śladu pozostawionego na szkle/papierze lub ze zdjęcia,
- **dziedziczne częściowo** — bliźnięta jednojajowe wykazują podobieństwa, ale nie identyczne wzorce.

### Historia i klasyfikacja Suzuki–Tsuchihashi (1970)

Pionierami systematycznego opisu typologii bruzd byli japońscy naukowcy **Suzuki i Tsuchihashi (1970)**, którzy na podstawie analizy setek odcisków ust sklasyfikowali bruzdy na **5 głównych typów**:

| Typ | Opis | Symbol |
|---|---|---|
| I | Bruzdy pionowe — przebiegają przez całą wargę | `|` |
| II | Bruzdy poziome — przebiegają prostopadle do osi wargi | `—` |
| III | Bruzdy rozgałęzione — dzielą się na gałęzie | `Y` |
| IV | Bruzdy krzyżujące — przecinają się wzajemnie | `X` |
| V | Bruzdy nieregularne — nie pasują do poprzednich typów | `~` |

Projekt nie implementuje tej typologii bezpośrednio (jest to możliwe rozszerzenie), lecz inspiruje się jej mechaniką: zamiast klasyfikować typ bruzdy, automatycznie wykrywa **punkty charakterystyczne (minucje)** metodą Crossing Number — tą samą, którą stosuje się w daktyloskopii.

### Zadanie identyfikacyjne

Projekt rozwiązuje zadanie **identyfikacji 1-do-N** (ang. *one-to-many*): dane jest jedno nieznane zdjęcie ust i szukamy, która spośród N=22 znanych osób jest jego autorem. Jest to trudniejsze zadanie niż **weryfikacja 1-do-1** (czy dwa zdjęcia należą do tej samej osoby), bo model musi jednocześnie uwzględniać wszystkie 22 klasy.

Losowy klasyfikator osiąga accuracy = 1/22 ≈ **4.5%** — jest to dolna granica odniesienia dla wyników eksperymentów.

---

## 2. Wymagania i instalacja

### Wymagania systemowe

- **Python 3.11 lub nowszy** — projekt używa składni typów (np. `X | Y`) dostępnej od 3.10+, oraz `match` i innych nowości 3.11.
- **System operacyjny: Windows** — skróty w sidebarze używają `os.startfile()` do otwierania folderów w Eksploratorze Windows. Na Linux/Mac ta funkcja nie istnieje (skróty będą nieaktywne, reszta aplikacji działa).
- **RAM: minimum 4 GB** — ekstrakcja Gabora i trening Random Forest (200 drzew) są pamięciochłonne. Przy 8 GB komfort jest znacznie większy.
- **Dysk: ~500 MB** na cache wektorów cech (`.npy`) dla 382 zdjęć.

### Instalacja

```bash
# Sklonuj repozytorium lub rozpakuj projekt
cd "c:/Magisterka/15 projekcik"

# Zainstaluj wszystkie zależności
pip install -r requirements.txt
```

### Plik `requirements.txt` — pełna lista

```
streamlit>=1.30
opencv-python>=4.9
scikit-image>=0.22
scikit-learn>=1.4
numpy>=1.26
pandas>=2.1
openpyxl>=3.1
matplotlib>=3.8
seaborn>=0.13
joblib>=1.3
albumentations>=1.4
```

> **Uwaga o albumentations**: biblioteka jest w requirements.txt ale **nie jest używana** w aktualnym pipeline — augmentacja danych została usunięta z projektu (zob. sekcja [19](#19-różnice-względem-pierwotnych-notatek-projektowych)). Można ją pominąć przy instalacji.

---

## 3. Uruchamianie aplikacji

### Polecenie startowe

```bash
cd "c:/Magisterka/15 projekcik"
streamlit run apps/app.py
```

Aplikacja uruchamia lokalny serwer HTTP i **automatycznie otwiera przeglądarkę** pod adresem `http://localhost:8502`. Port 8502 jest skonfigurowany w `.streamlit/config.toml` (domyślny Streamlit to 8501 — zmieniono aby uniknąć konfliktów z innymi instancjami).

### Nawigacja — 8 stron w 2 sekcjach

```
Metoda 2: Klasyczne ML
  1. Eksploracja danych          (/eksploracja)
  2. Preprocessing               (/preprocessing)
  3. Ekstrakcja cech             (/features)
  4. Trening modelu              (/training)
  5. Predykcja                   (/prediction)
  6. Ewaluacja                   (/evaluation)

Metoda 1: Tradycyjna
  7. Predykcja (obraz–obraz)     (/m1_prediction)
  8. Ewaluacja (CV)              (/m1_evaluation)
```

Każda strona jest osobnym modułem Python (`render()` funkcja). Nawigacja zaimplementowana przez `st.navigation()` z `st.Page()` — nowy mechanizm wielostronicowości Streamlit (≥1.30), który zastąpił starszy system plików w katalogu `pages/`.

### Zalecana kolejność (pierwsze uruchomienie)

1. **Strona 1** — przejrzyj statystyki bazy danych (czy załadowała się poprawnie)
2. **Strona 4** — **wytrenuj modele** (WYMAGANE przed stronami 5 i 6)
3. **Strona 5** — przetestuj predykcję na kilku zdjęciach
4. **Strona 6** — uruchom pełną ewaluację 5-fold CV i wyeksportuj wyniki

Strony 2 i 3 są pomocnicze (wizualizacja kroków pipeline) — można pominąć przy pierwszym uruchomieniu.

---

## 4. Struktura projektu

```
15 projekcik/
│
├── apps/                                  ← Warstwa GUI (Streamlit)
│   ├── app.py                             ← GŁÓWNY punkt wejścia — uruchamiaj TEN
│   ├── sidebar_shortcuts.py               ← Skróty do folderów w pasku bocznym
│   ├── method1_pages/
│   │   ├── __init__.py
│   │   ├── _shared.py                     ← Cached dataset + metadane dla stron M1
│   │   ├── page_prediction.py             ← Strona 7 — predykcja Metody 1
│   │   └── page_evaluation.py             ← Strona 8 — ewaluacja CV Metody 1
│   └── method2_pages/
│       ├── __init__.py
│       ├── page_01_eksploracja.py         ← Strona 1
│       ├── page_02_preprocessing.py       ← Strona 2
│       ├── page_03_features.py            ← Strona 3
│       ├── page_04_training.py            ← Strona 4
│       ├── page_05_prediction.py          ← Strona 5
│       └── page_06_evaluation.py          ← Strona 6
│
├── src/                                   ← Logika biznesowa (bez GUI)
│   ├── core/                              ← Moduły wspólne dla obu metod
│   │   ├── config.py                      ← Ścieżki i stałe (jeden plik dla całego projektu)
│   │   ├── filename_parser.py             ← Parser nazw plików PNG
│   │   ├── dataset_loader.py              ← Klasa LipDataset
│   │   ├── metadata_loader.py             ← Parser pliku .xlsm z metadanymi
│   │   ├── splits.py                      ← 5-fold CV, zapis/odczyt JSON
│   │   ├── dataset_hash.py                ← SHA-256 hash folderu (auto-retrain)
│   │   └── results_writer.py              ← Zapis wyników do .xlsx
│   ├── method1_traditional/
│   │   ├── ssim_compare.py                ← SSIM score
│   │   ├── orb_compare.py                 ← ORB keypoint matching score
│   │   ├── histogram_compare.py           ← Histogram correlation score
│   │   └── classifier.py                  ← Klasyfikator 1-NN (TraditionalClassifier)
│   └── method2_ml/
│       ├── preprocessing.py               ← run_pipeline() → PreprocessingResult
│       ├── features_lbp.py                ← Local Binary Patterns
│       ├── features_hog.py                ← Histogram of Oriented Gradients
│       ├── features_gabor.py              ← Bank filtrów Gabora
│       ├── features_minutiae.py           ← Binaryzacja + szkielet + minucje
│       ├── feature_vector.py              ← FeatureExtractorPipeline (PCA + scaler)
│       └── classifier.py                  ← LipClassifier (SVM / RF / k-NN)
│
├── data/
│   ├── baza_danych/                       ← 382 zdjęć PNG — NIE MODYFIKOWAĆ
│   ├── 10 osoby - ok.xlsm                 ← Metadane 22 osób
│   ├── splits/                            ← Pliki JSON z podziałem foldów (auto)
│   │   ├── fold_0.json ... fold_4.json
│   └── processed/                         ← Cache wektorów cech (auto)
│       └── <nr_osoby>/<stem>/features.npy
│
├── models/                                ← Wytrenowane modele (po treningu)
│   ├── method2_svm.joblib
│   ├── method2_rf.joblib
│   ├── method2_knn.joblib
│   ├── method2_pipeline.joblib            ← scaler + PCA (do predykcji)
│   └── dataset_hash.txt                   ← Hash bazy przy ostatnim treningu
│
├── results/                               ← Wyniki ewaluacji
│   ├── experiment_<timestamp>.xlsx
│   └── plots/
│       └── confusion_matrix_<clf>.png
│
├── exports/                               ← Szczegółowy log predykcji z GUI
│   └── predykcje_szczegolowe.xlsx         ← 78 kolumn per predykcja
│
├── .streamlit/
│   └── config.toml                        ← Port, motyw kolorów
│
├── notatki/
│   └── 01_początek.md                     ← Pierwotne notatki projektowe
│
├── requirements.txt
└── README.md
```

### Zasada separacji warstw

- **`src/`** — czysta logika biznesowa. Żadnych importów `streamlit`. Można używać z linii poleceń, testów jednostkowych, notebooków Jupyter.
- **`apps/`** — wyłącznie warstwa prezentacji. Importuje z `src/`, nie zawiera algorytmów.
- **`data/`** — dane wejściowe i cache. Folder `baza_danych/` jest tylko do odczytu przez aplikację.
- **`models/`** — artefakty treningowe. Generowane przez aplikację, wczytywane przy predykcji.

---

## 5. Dane wejściowe

### Baza zdjęć (`data/baza_danych/`)

- **382 zdjęcia PNG**
- Rozdzielczość oryginalna: **1478×560 px** (szerokość × wysokość)
- Format: PNG (bezstratny) — ważne dla jakości ekstrakcji cech
- **22 osoby**, średnio ~17 zdjęć na osobę (rozkład nierównomierny)
- Każde zdjęcie przedstawia **wycięty obszar ust** (`_CUT` w nazwie) — detekcja ust nie jest potrzebna
- Zdjęcia wykonane w dwóch wariantach oświetlenia: **LampaON** (z dodatkowym doświetleniem) i **LampaOFF** (naturalne/otoczenie)
- Zdjęcia ON i OFF są **wymieszane** w zbiorach treningowych i testowych (brak podziału per lampa) — model musi być odporny na obie wersje

### Metadane osób (`data/10 osoby - ok.xlsm`)

Plik Microsoft Excel z obsługą makr (`.xlsm`), zawierający:
- Arkusze `01` do `22` — każdy to dane jednej osoby
- Arkusz `Dane` — słowniki referencyjne (pomijany przy wczytywaniu)

Każdy arkusz osoby zawiera następujące pola (wczytywane z wiersza 4, po nagłówkach PL i EN):

| Pole | Typ | Opis i znaczenie dla badania |
|---|---|---|
| `gender` | tekst | Płeć — do analizy czy płeć wpływa na accuracy |
| `age` | liczba | Wiek — do analizy zmienności z wiekiem |
| `skin_tone` | tekst | Odcień skóry — do analizy czy ciemniejsza skóra obniża accuracy (kontrast bruzd) |
| `unique_characteristics` | tekst | Cechy szczególne ust (blizny, pieprzyki itp.) |
| `lipstick` | bool | Czy osoba nosi szminki — może zaburzać strukturę bruzd |
| `lip_balm` | bool | Czy używa balsamu — może wygładzać bruzdy |
| `facial_hair_interference` | bool | Czy zarost wchodzi w kadr i zakłóca obraz |
| `excessive_reflections` | bool | Czy widoczne nadmierne refleksy od lampy |

**Techniczny szczegół wczytywania**: plik `.xlsm` jest wczytywany przez `openpyxl` z parametrami `data_only=True` (zwraca wartości komórek, nie formuły) i `keep_vba=False` (ignoruje makra VBA — przyspiesza wczytywanie). Iteracja po arkuszach `'01'` do `'22'` (ze stringowym kluczem z zerem wiodącym).

---

## 6. Moduły wspólne — `src/core/`

### `config.py` — centralna konfiguracja

Jedyne miejsce w projekcie, gdzie zdefiniowane są ścieżki i stałe. Wszystkie inne moduły importują stąd, nigdy nie używają ścieżek zakodowanych na stałe.

```python
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # wyznaczane dynamicznie

DATA_DIR        = PROJECT_ROOT / "data"
BAZA_DANYCH_DIR = DATA_DIR / "baza_danych"
METADATA_XLSM   = DATA_DIR / "10 osoby - ok.xlsm"
PROCESSED_DIR   = DATA_DIR / "processed"
SPLITS_DIR      = DATA_DIR / "splits"
MODELS_DIR      = PROJECT_ROOT / "models"
RESULTS_DIR     = PROJECT_ROOT / "results"
PLOTS_DIR       = RESULTS_DIR / "plots"
EXPORTS_DIR     = PROJECT_ROOT / "exports"

RANDOM_SEED = 42        # używany wszędzie dla reprodukowalności
N_SPLITS    = 5         # liczba foldów CV
TARGET_SIZE = (512, 256)  # (szerokość, wysokość) po preprocessingu
```

Funkcja `ensure_dirs()` tworzy wszystkie foldery wyjściowe jeśli nie istnieją — wywoływana przy starcie aplikacji.

---

### `filename_parser.py` — dekodowanie nazw plików

Każde zdjęcie w bazie nosi nazwę zakodowaną z metadanymi. Parser rozbija nazwę na komponenty.

**Format nazwy:**
```
2026-03-05_08-25-34_Nr_2_LampaOFF_Fokus_0_68_Exp_0_CUT.png
```

**Kluczowy problem do rozwiązania — niespójność numeracji:** w bazie danych część plików ma `Nr_01` (z zerem wiodącym), a część `Nr_2` (bez zera). Parser **normalizuje zawsze do czystego `int`**: `Nr_01 → 1`, `Nr_2 → 2`. Bez tej normalizacji dwie osoby byłyby mylone jako różne.

Zwracane pola przez `parse_filename()`:

| Pole | Typ | Przykład | Opis |
|---|---|---|---|
| `date` | str | `"2026-03-05"` | Data wykonania zdjęcia |
| `time_str` | str | `"08-25-34"` | Godzina (separatory `-`) |
| `person_id` | int | `2` | Numer osoby (znormalizowany) |
| `flash` | str | `"OFF"` | Stan lampy: `"ON"` lub `"OFF"` |
| `focus` | float | `0.68` | Fokus aparatu |
| `exposure` | int | `0` | Ekspozycja (może być ujemna) |
| `is_cut` | bool | `True` | Czy zdjęcie jest wycięte |

Funkcja `try_parse_filename()` zwraca `None` dla plików o niestandardowej nazwie (np. pliki zewnętrzne przesłane przez upload z GUI) — bez rzucania wyjątku.

---

### `dataset_loader.py` — zarządzanie bazą zdjęć

Klasa `LipDataset` enkapsuluje całą logikę dostępu do bazy zdjęć.

```python
dataset = LipDataset(BAZA_DANYCH_DIR)

# Listowanie
dataset.list_persons()          # → [1, 2, 3, ..., 22]  (List[int])
dataset.list_images(person_id)  # → [Path, Path, ...]    (pliki PNG danej osoby)
dataset.total_count()           # → 382                  (łączna liczba zdjęć)
dataset.iterate_all()           # → Generator: (person_id, path, image_bgr)

# Ładowanie obrazu — jedyne miejsce w projekcie
LipDataset.load_image(path, color="rgb")   # → np.ndarray (H×W×3), uint8
LipDataset.load_image(path, color="bgr")   # → np.ndarray (H×W×3), uint8
LipDataset.load_image(path, color="gray")  # → np.ndarray (H×W), uint8
```

`load_image()` jest metodą statyczną — można jej użyć bez instancji klasy. Używana przez wszystkie moduły w projekcie jako jedyny punkt wejścia dla wczytywania plików PNG.

---

### `metadata_loader.py` — wczytywanie metadanych Excel

```python
from src.core.metadata_loader import load_person_metadata

metadata = load_person_metadata(METADATA_XLSM)
# → Dict[int, PersonMeta]  (klucz: person_id 1–22)

meta = metadata[5]          # metadane osoby Nr_05
meta.gender                  # "female"
meta.age                     # 24
meta.skin_tone               # "jasna"
meta.unique_characteristics  # "brak"
```

`PersonMeta` to dataclass z polami opisanymi w sekcji 5. Iteracja po arkuszach `'01'` do `'22'` (stringowe klucze) — arkusz `'Dane'` jest pomijany.

---

### `splits.py` — podział na foldy Cross-Validation

#### Czym jest Stratified K-Fold?

**K-Fold Cross-Validation** to technika walidacji, w której zbiór danych jest dzielony na K równych części (foldów). Model jest trenowany K razy — za każdym razem jeden fold służy jako zbiór testowy, pozostałe K-1 jako treningowy. Wynikowa metryka to średnia z K przebiegów.

**Stratified** oznacza, że proporcje klas są zachowane w każdym foldzie. Bez stratyfikacji mogłoby się zdarzyć, że jedna osoba ma 0 zdjęć w zbiorze testowym danego folda. Przy 22 osobach i ~17 zdjęciach/osobę, każdy fold testowy powinien zawierać ~3–4 zdjęcia per osoba.

#### Implementacja

```python
from src.core.splits import get_or_create_splits, Fold

folds = get_or_create_splits(dataset)  # → List[Fold], długość 5
```

Wewnętrznie używa `sklearn.model_selection.StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.

Dataclass `Fold`:

```python
fold.fold_index     # 0, 1, 2, 3 lub 4
fold.train_paths    # ~306 ścieżek (jako stringi, serializowalne do JSON)
fold.test_paths     # ~76 ścieżek
fold.train_labels   # odpowiadające etykiety (person_id)
fold.test_labels
fold.n_train        # len(train_paths)
fold.n_test         # len(test_paths)
```

Raz obliczone foldy są zapisywane do `data/splits/fold_0.json` ... `fold_4.json`. Przy kolejnym wywołaniu `get_or_create_splits()` pliki są **wczytywane z JSON** — gwarantuje to identyczne podziały między sesjami, maszynami i użytkownikami (reprodukowalność wyników naukowych).

---

### `dataset_hash.py` — wykrywanie zmian w bazie

Mechanizm pozwalający wykryć, że baza danych uległa zmianie od czasu ostatniego trenowania modelu.

#### Jak obliczany jest hash?

Dla każdego pliku w `data/baza_danych/` pobierana jest krotka `(nazwa_pliku, rozmiar_bajtów, mtime)`. Lista krotek jest sortowana leksykograficznie, łączona w jeden string i szyfrowana SHA-256.

Dlaczego tak, a nie przez liczenie sumy kontrolnej treści plików? Bo przy 382 plikach PNG (~kilkaset MB) odczytanie treści byłoby wolne. Metoda `(nazwa + rozmiar + mtime)` jest lekka i wystarczająco unikalna dla praktycznych zastosowań.

```python
hash_value = compute_dataset_hash()         # → str (hex SHA-256)
save_current_hash(hash_value)               # → zapisuje do models/dataset_hash.txt
saved = load_saved_hash()                   # → str | None
needs_retrain = is_retrain_needed()         # → bool
```

---

### `results_writer.py` — eksport wyników

Dwa niezależne mechanizmy eksportu:

#### 1. `ExperimentResults` — wyniki ewaluacji 5-fold CV

```python
experiment = ExperimentResults()

# Dodawanie danych
experiment.add_run_info("HOG_PCA_components", "100")
experiment.add_fold_summary(FoldSummary(fold=1, method="svm", accuracy=0.82, ...))
experiment.add_prediction(PredictionRow(image_path=..., true_id=5, ...))
experiment.add_confusion_matrix("svm", cm_array_22x22, class_labels)

# Zapis
path = experiment.save()  # → results/experiment_2026-04-13_14-30-00.xlsx
```

Generowane arkusze:

| Arkusz | Co zawiera |
|---|---|
| `Run_info` | Timestamp, parametry uruchomienia (HOG_PCA, klasyfikatory, N_folds) |
| `Predictions` | Każda predykcja z CV: plik, true_id, method, predicted_id, top3, top5, czas |
| `Summary_svm` / `Summary_rf` / `Summary_knn` | Metryki per fold + mean ± std |
| `ConfusionMatrix_svm` itd. | Macierz 22×22 zsumowana z 5 foldów |

#### 2. `write_detailed_prediction_excel()` — log predykcji z GUI

Funkcja do zapisu/dopisywania wiersza do `exports/predykcje_szczegolowe.xlsx`. Każde kliknięcie "Zapisz do Excela" na Stronie 5 dopisuje **jeden wiersz** (78 kolumn).

Przy pierwszym wywołaniu plik jest tworzony z kolorowym arkuszem **Legenda** (opis każdej z 78 kolumn: nazwa, opis, jednostka/zakres).

78 kolumn podzielonych na grupy:

| Grupa | Liczba kolumn | Zawartość |
|---|---|---|
| Meta | 5 | Plik, osoba_prawdziwa, lampa, data, typ_zdjecia (z_bazy/zewnetrzne) |
| Preprocessing | 13 | Rozmiary, jasność oryginału i po preprocessingu, parametry CLAHE i bilateral |
| Cechy LBP | 11 | 10 binów histogramu + entropia |
| Cechy HOG | 2 | Energia + orientacja dominująca |
| Cechy Gabor | 3 | Średnia, maks. odpowiedź, orientacja dominująca |
| Cechy Minucji | 7 | n_endings, n_bifurcations, n_crossings, n_total, gęstość, piksele_szkieletu, entropia_binarna |
| Wyniki SVM | 11 | Top-1–5 z confidence + TAK/NIE |
| Wyniki RF | 11 | Top-1–5 z confidence + TAK/NIE |
| Wyniki k-NN | 11 | Top-1–5 z confidence + TAK/NIE |
| Podsumowanie | 3 | ile_clf_zgodnych, konsensus_top1, czas_total_ms, timestamp |

---

## 7. Metoda 1 — tradycyjne porównanie obrazów

### Zasada działania

Brak ekstrakcji cech. Dla każdego zdjęcia testowego obliczamy miarę podobieństwa do **wszystkich** zdjęć treningowych i wybieramy etykietę osoby z najbardziej podobnego zdjęcia — jest to klasyfikator **1-Nearest Neighbor (1-NN)** w przestrzeni score'ów podobieństwa.

Zalety: prosta implementacja, brak fazy treningowej (poza załadowaniem zdjęć do pamięci), interpretowalne wyniki (widać które zdjęcie było najbardziej podobne).

Wady: wolne przy dużych bazach (każde porównanie liniowe po wszystkich treningowych), wrażliwe na zmiany oświetlenia i rotacji.

---

### `ssim_compare.py` — Structural Similarity Index

**Co to jest SSIM?** SSIM (Wang et al., 2004) to miara podobieństwa obrazów, która modeluje **percepcję ludzką** — zamiast porównywać piksele wprost (jak MSE), porównuje trzy właściwości lokalne:

1. **Luminancja** (jasność): `l(x,y) = (2μx·μy + C1) / (μx² + μy² + C1)`
2. **Kontrast**: `c(x,y) = (2σx·σy + C2) / (σx² + σy² + C2)`
3. **Struktura** (korelacja lokalnych odchyleń): `s(x,y) = (σxy + C3) / (σx·σy + C3)`

Końcowy wynik: `SSIM(x,y) = l(x,y)^α · c(x,y)^β · s(x,y)^γ` ∈ [-1, 1]

Gdzie `μ` to lokalna średnia, `σ²` to wariancja, `σxy` to kowariancja, a `C1, C2, C3` to małe stałe stabilizujące.

Wynik 1.0 = obrazy identyczne. Wynik 0 = brak korelacji. Wartości ujemne są możliwe (obrazy odwrócone). Wyższe = bardziej podobne.

**Implementacja**: oba obrazy skalowane do **256×128 px** (grayscale) przed porównaniem — rozmiar wystarczający do zachowania struktury bruzd, wielokrotnie szybszy niż porównanie pełnych 1478×560 px.

---

### `orb_compare.py` — ORB Keypoint Matching

**Co to jest ORB?** ORB (Oriented FAST and Rotated BRIEF, Rublee et al., 2011) to szybki detektor i deskryptor punktów kluczowych (ang. keypoints). Łączy:

- **FAST** (Features from Accelerated Segment Test) — szybka detekcja rogów obrazu przez porównanie jasności piksela z okrągiem sąsiadów
- **BRIEF** (Binary Robust Independent Elementary Features) — binarny deskryptor (ciąg 0/1) opisujący otoczenie keypointa przez losowe porównania parami

**Jak działa dopasowanie:**
1. Dla obu obrazów wykrywane są keypoints (`cv2.ORB_create(nfeatures=500)`)
2. Obliczane są deskryptory binarne (256-bitowe wektory)
3. Dopasowanie przez `cv2.BFMatcher` z `cv2.NORM_HAMMING` (odległość Hamminga — liczba różnych bitów)
4. Filtrowanie przez **test Lowe'a** (ratio test): dobre dopasowanie = odległość do 1. sąsiada / odległość do 2. sąsiada < 0.75
5. Score = liczba dobrych dopasowań / max(n_keypoints_query, n_keypoints_train)

Zaleta ORB nad SIFT/SURF: jest szybszy i nie ma licencji patentowych. Działa dobrze dla zdjęć z wyraźną strukturą (bruzdy ust są dobrym kandydatem).

---

### `histogram_compare.py` — Histogram Correlation

Porównanie rozkładów jasności pikseli (histogramów).

**Implementacja:**
1. Konwersja obu obrazów do grayscale
2. Obliczenie histogramu 256-binowego dla każdego (`cv2.calcHist`)
3. Normalizacja (suma = 1)
4. Porównanie przez `cv2.compareHist` z metodą `HISTCMP_CORREL`

**Metoda korelacji Pearsona** na histogramach:
```
correl(H1, H2) = Σ(H1_i - H1_mean)(H2_i - H2_mean) / sqrt(Σ(H1_i-H1_mean)² · Σ(H2_i-H2_mean)²)
```
Wynik ∈ [-1, 1]. Dwa obrazy o identycznych rozkładach jasności → 1.0. Niezależne rozkłady → 0.

Histogram jest prostą, globalną cechą — nie uwzględnia układu przestrzennego pikseli. Dlatego jest najsłabszą z trzech miar osobno, ale wzmacnia Combined.

---

### Miara Combined (ważona kombinacja)

Łączy SSIM, ORB i histogram w jedno score:

```python
combined = 0.50 × (SSIM+1)/2 + 0.25 × ORB + 0.25 × (hist+1)/2
```

SSIM i histogram normalizowane z [-1,1] → [0,1] przez `(x+1)/2`. ORB jest już w [0,1].

Wagi `0.5 : 0.25 : 0.25` — SSIM dostaje największy udział bo jest najbardziej informacyjny strukturalnie. Empirycznie sprawdzono że Combined daje lepsze wyniki niż każda miara osobno.

---

### `classifier.py` — `TraditionalClassifier`

```python
clf = TraditionalClassifier()
clf.fit(train_paths, train_labels)     # ładuje obrazy RGB do pamięci

result = clf.predict(query_img, method="ssim", top_k=5)
# result['predicted']  → person_id (int) — top-1
# result['confidence'] → float — score top-1
# result['top_k']      → [(person_id, score, train_idx), ...]
# result['all_scores'] → np.ndarray — score per każde zdjęcie treningowe

metrics = clf.score(query_paths, query_labels, method="combined")
# metrics['accuracy'] → float
# metrics['top3']     → float
# metrics['top5']     → float
# metrics['predictions'] → lista słowników per predykcja
```

---

## 8. Metoda 2 — ML na cechach biometrycznych

### Ogólna idea

Zamiast porównywać obrazy bezpośrednio, wyciągamy z każdego zdjęcia **numeryczny wektor cech** — skondensowaną reprezentację biometryczną (181 liczb zamiast 512×256 = 131 072 pikseli). Następnie klasyfikator ML uczy się rozróżniać 22 osoby w tej 181-wymiarowej przestrzeni cech.

Wektor cech = konkatenacja 4 ekstraktorów:
```
LBP (10) + HOG→PCA (100) + Gabor (64) + Minucje (7) = 181 cech
```

---

### Pipeline preprocessingu — `preprocessing.py`

Preprocessing przekształca surowy obraz BGR (1478×560 px) w zunifikowany obraz grayscale (512×256 px) odpowiedni do ekstrakcji cech.

**Krok 1: Konwersja do skali szarości**

`cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` — eliminuje informację o kolorze (bruzdy ust są strukturą geometryczną, nie kolorową). Redukuje dane 3× (3 kanały → 1 kanał).

**Krok 2: CLAHE** (Contrast Limited Adaptive Histogram Equalization)

Standardowe wyrównanie histogramu (HE) rozciąga kontrast globalnie — może prześwietlać jasne obszary i zbyt ciemnieć ciemne. CLAHE działa **lokalnie**:
- Dzieli obraz na siatkę kafelków (domyślnie 8×8 kafelków, każdy 64×32 px przy docelowym rozmiarze)
- W każdym kafelku wyrównuje histogram oddzielnie
- **Clip limit** (domyślnie 2.0) ogranicza maksymalną amplifikację kontrastu — bez tego CLAHE wzmacniałby też szum
- Interpolacja bilinearna między kafelkami eliminuje artefakty na granicach

Rezultat: bruzdy ust stają się wyraźniejsze i bardziej jednorodne niezależnie od oryginalnego oświetlenia (lampa ON vs OFF).

**Krok 3: Bilateral Filter**

Standardowe rozmycia (Gaussian blur) wygładzają krawędzie razem z szumem. Bilateral filter zachowuje krawędzie bo **waży sąsiadów podwójnie** — według odległości przestrzennej i podobieństwa koloru:

```
BF[I]_p = (1/W_p) Σ_q f(||p-q||) · g(|I_p - I_q|) · I_q
```

Gdzie `f` to jądro przestrzenne (Gaussian), `g` to jądro koloru (Gaussian). Piksele podobnej jasności co środkowy mają duży wpływ (przechodzą filtr), piksele bardzo różne jasności (krawędź) mają mały wpływ (nie zamazują krawędzi).

Parametry: `d=9` (średnica sąsiedztwa), `sigma_color=75` (szerokość jądra koloru), `sigma_space=75` (szerokość jądra przestrzennego).

**Krok 4: Resize do 512×256 px**

`cv2.resize(..., interpolation=cv2.INTER_AREA)` — interpolacja AREA jest najlepsza przy zmniejszaniu (antialiasinguje przez uśrednianie). Unified rozmiar jest konieczny by wektory cech wszystkich obrazów miały identyczną długość.

Funkcja `run_pipeline()` zwraca dataclass `PreprocessingResult` z polami `original_bgr`, `grayscale`, `clahe`, `denoised`, `resized` oraz użytymi parametrami — każdy krok dostępny do wyświetlenia w GUI.

---

### Ekstrakcja cech

#### `features_lbp.py` — Local Binary Patterns

**Koncepcja:** LBP (Ojala et al., 2002) to deskryptor tekstury. Dla każdego piksela obrazu porównuje jego jasność z P sąsiadami w promieniu R i koduje wynik binarnie:
```
LBP(p) = Σ_{i=0}^{P-1} s(g_i - g_c) · 2^i
```
Gdzie `g_c` to jasność piksela centralnego, `g_i` to jasność i-tego sąsiada, `s(x) = 1` jeśli x≥0, `0` jeśli x<0.

**Wariant `uniform`:** Wzorzec LBP jest "uniform" jeśli ma co najwyżej 2 przejścia 0↔1 w binarnej reprezentacji (obchodząc cyklicznie). Dla P=8 istnieje 58 takich wzorców + 1 bin zbiorczy "non-uniform". Wzorce uniform odpowiadają strukturom krawędziowym i narożnikowym.

**Parametry projektu:** P=8, R=1, method='uniform', histogram 10-binowy.

Wynik: histogram 10-binowy znormalizowany (suma=1). 10 binów to kompresja 59 możliwych wartości — arbitralna, ale wystarczająca do opisu globalnej tekstury obszaru.

- Bin 0 ≈ udział pikseli "jednorodnie ciemnych" (tło)
- Bin 9 ≈ udział pikseli "jednorodnie jasnych" (jasne obszary)
- Biny środkowe ≈ udział różnych wzorców krawędziowych

**Wektor cech: 10 wartości.**

---

#### `features_hog.py` — Histogram of Oriented Gradients

**Koncepcja:** HOG (Dalal & Triggs, 2005) opisuje lokalną strukturę krawędzi przez histogramy kierunków gradientu. Pierwotnie stworzony do detekcji pieszych, doskonale opisuje też strukturę liniową bruzd ust.

**Kroki algorytmu:**

1. **Obliczenie gradientów:** dla każdego piksela liczymy gradient poziomy i pionowy (filtr [-1, 0, 1]), następnie magnitudę `m = sqrt(Gx² + Gy²)` i orientację `θ = arctan(Gy/Gx)`.

2. **Podział na komórki:** obraz 512×256 dzielony na komórki 16×16 px → siatka 32×16 = 512 komórek.

3. **Histogram orientacji per komórka:** w każdej komórce budowany jest histogram 9-binowy orientacji (0°–180° podzielone na 9 przedziałów po 20°). Gradiendy ważone magnitudą.

4. **Normalizacja blokami:** komórki grupowane w bloki 2×2 (z zakładką). Histogram bloku (4 komórki × 9 = 36 wartości) normalizowany L2-Hys (norma L2, clip do 0.2, renormalizacja L2). Liczba bloków: (32-1)×(16-1) = 465.

5. **Konkatenacja:** 465 bloków × 36 = **16 740 cech**.

**PCA:** 16 740 cech dla 382 próbek to sytuacja "więcej cech niż próbek" — klasyczne **przekleństwo wymiarowości** (ang. curse of dimensionality). Klasyfikatory liniowe i SVM radzą sobie z tym przez regularyzację, ale RF i k-NN cierpią. PCA redukuje do 100 składowych zachowując >95% wariancji.

**Obliczenie wymiarów:**
```
Komórki:  (512÷16) × (256÷16) = 32 × 16 = 512
Bloki:    (32-1) × (16-1)      = 31 × 15 = 465
Cech/blok: 2 × 2 × 9           = 36
Łącznie:   465 × 36            = 16 740
```

**Wektor cech (po PCA): 100 wartości.**

---

#### `features_gabor.py` — Bank filtrów Gabora

**Koncepcja:** Filtr Gabora to iloczyn funkcji gaussowskiej i fali sinusoidalnej — w przestrzeni częstotliwości jest to bandpass filter o określonej częstotliwości i orientacji. Biologicznie odpowiada receptorom kory wzrokowej V1 reagującym na krawędzie pod określonym kątem.

Dla biometrii ust: bruzdy przebiegają w różnych kierunkach (pionowo, poziomo, ukośnie). Bank filtrów Gabora wychwytuje energię sygnału w każdym kierunku osobno, dając bogatą charakterystykę tekstury kierunkowej.

**Parametry jądra (`cv2.getGaborKernel`):**

```
kernel_size = 21×21 px     — rozmiar maski splotu
sigma       = 4.0          — sigma gaussowskiej koperty (szerokość odpowiedzi)
theta       = i·π/8        — orientacja filtra (i = 0..7, co 22.5°)
lambda      = 1/freq       — długość fali nośnej
gamma       = 0.5          — współczynnik eliptyczności (stosunek osi)
psi         = 0            — faza
```

Częstotliwości: `(0.1, 0.2, 0.3, 0.4)` cycles/pixel. Niska częstotliwość (0.1) wychwytuje grube, szerokie bruzdy. Wysoka (0.4) wychwytuje cienkie, drobne szczegóły.

**Dla każdego filtra:**
- Splot obrazu z jądrem: `response = |cv2.filter2D(img, CV_32F, kernel)|` (energia)
- Dwie cechy: `mean(|response|)` i `std(|response|)`

32 filtry × 2 statystyki = **64 wartości**.

Bank filtrów jest budowany raz globalnie (`_DEFAULT_BANK`) — budowanie jąder jest kosztowne obliczeniowo, warto to robić raz.

---

#### `features_minutiae.py` — Minucje bruzd ust

Analogia do daktyloskopii: tak jak linie papilarne mają charakterystyczne punkty (zakończenia, rozwidlenia), tak bruzdy ust mają analogiczne struktury.

**Krok 1: Binaryzacja** — oddzielenie bruzd od tła

Trzy metody do wyboru (przez GUI na Stronie 3):

| Metoda | Opis | Parametry | Kiedy używać |
|---|---|---|---|
| `adaptive` (domyślna) | `cv2.adaptiveThreshold` z metodą Gaussian | block_size=35, C=10 | Nierównomierne oświetlenie, ogólny przypadek |
| `otsu` | Globalny próg Otsu | — | Równomierne oświetlenie, szybszy |
| `gabor_otsu` | Wzmocnienie Gaborem (θ=45°) + Otsu | sigma=4, lambda=8 | Najlepsza jakość, wolniejszy |

Adaptive threshold wyznacza próg lokalnie dla każdego bloku 35×35 px (minus stała C=10). Dzięki temu bruzdy są dobrze widoczne nawet jeśli jedna część obrazu jest jaśniejsza.

**Krok 2: Skeletonizacja** — `skimage.morphology.skeletonize`

Algorytm iteracyjnie usuwa piksele z brzegów bruzd binarnych zachowując połączenia — redukuje bruzdy do linii o szerokości **1 piksel**. Dzięki temu można precyzyjnie liczyć sąsiedztwo każdego piksela.

**Krok 3: Crossing Number** — detekcja punktów charakterystycznych

Dla każdego piksela szkieletu (`= 1`) obliczamy Crossing Number:

```
CN = 0.5 × Σ|p_i - p_{i+1}|   (i = 0..7, cyklicznie)
```

Gdzie `p_0..p_7` to piksele sąsiedztwa 3×3 w kolejności zgodnej z ruchem wskazówek zegara:
```
p2 p3 p4
p1  c p5
p8 p7 p6
```

Znaczenie wartości CN:

| CN | Typ | Znaczenie |
|---|---|---|
| 1 | **Ending** (zakończenie) | Bruzda się tu kończy |
| 2 | Kontynuacja | Piksel środka bruzdy — nie jest minucją |
| 3 | **Bifurcation** (rozwidlenie) | Bruzda się tu rozgałęzia |
| ≥4 | **Crossing** (skrzyżowanie) | Dwie bruzdy się tu przecinają |

**Krok 4: Wektor cech globalnych**

```python
feature_vector() → np.ndarray([
    n_endings,           # liczba zakończeń (CN=1)
    n_bifurcations,      # liczba rozwidleń (CN=3)
    n_crossings,         # liczba skrzyżowań (CN≥4)
    n_total_minutiae,    # suma powyższych
    groove_density,      # n_total / 10000 pikseli szkieletu
    total_skeleton_pixels,  # całkowita długość bruzd
    binary_entropy,      # entropia Shannona obrazu binarnego (0–1)
])
```

**Wektor cech: 7 wartości.**

---

### `feature_vector.py` — pipeline łączący wszystkie cechy

#### Architektura `FeatureExtractorPipeline`

```
train_paths → extract_raw_features() (z cache .npy)
            → X_raw (n × 16821)
            → PCA na HOG (16740 → 100)
            → konkatenacja: LBP(10) + HOG_pca(100) + Gabor(64) + Minutiae(7) = 181
            → StandardScaler.fit_transform()
            → X_train (n × 181), zestandaryzowany
```

Przy predykcji (transform, nie fit):
```
test_paths → extract_raw_features()
           → PCA.transform() (już wytrenowana na train)
           → konkatenacja
           → StandardScaler.transform() (już wytrenowany na train)
           → X_test (m × 181)
```

#### Cache wektorów cech

Ekstrakcja surowych cech dla jednego zdjęcia zajmuje ~0.15 s. Przy 382 zdjęciach i 5 foldach to ~300 sekund per fold. Dlatego surowy wektor cech (16 821 float32 = ~66 KB) jest cache'owany:

```
data/processed/<nr_osoby>/<stem_pliku>/features.npy
```

Cache jest **inwalidowany automatycznie** gdy plik `.npy` jest starszy niż plik źródłowy `.png`. Przy kolejnych uruchomieniach ekstrakcja spada z ~90 s do ~2 s dla całej bazy.

#### StandardScaler

**Po co skalować?** Różne ekstraktory mają różne skale: n_endings to liczba całkowita (0–500), LBP bins to ułamki (0.0–1.0), energię Gabora to wartości float (0–10 000). Bez skalowania cechy o dużych wartościach dominowałyby w metrykach odległości (k-NN, SVM z jądrem RBF). StandardScaler normalizuje każdą cechę do `mean=0, std=1`:

```
x_scaled = (x - mean_train) / std_train
```

`mean_train` i `std_train` są obliczane wyłącznie na zbiorze **treningowym** każdego folda i zastosowane (bez ponownego fitowania) do zbioru testowego — zapobiega to wyciekowi informacji z testu do treningu.

---

### `classifier.py` — klasyfikatory `LipClassifier`

Klasa `LipClassifier` opakowuje trzy klasyfikatory z jednolitym API — każdy ma takie same metody `fit()`, `predict_top1()`, `predict_proba()`, `predict_single()`, `save()`, `load()`.

#### SVM — Support Vector Machine

**Idea:** SVM szuka hiperpłaszczyzny w przestrzeni cech, która **maksymalizuje margines** między klasami. W przypadku wieloklasowym (22 klasy) sklearn używa strategii **OvO** (One-vs-One): trenuje C(22,2) = 231 binarnych klasyfikatorów, wynik to klasa z największą liczbą "zwycięstw".

**Jądro RBF** (Radial Basis Function): `K(x,y) = exp(-γ||x-y||²)`. Mapuje dane do nieskończonowymiarsowej przestrzeni — pozwala na nieliniowe granice decyzyjne bez jawnego obliczania tej przestrzeni ("kernel trick").

Parametry:
- `C=10.0` — parametr regularyzacji (kara za błędy klasyfikacji). Duże C = mniejszy margines, mniej błędów na treningu. Wartość 10 oznacza umiarkowaną regularyzację.
- `gamma='scale'` — szerokość jądra RBF. `'scale'` = 1/(n_features × Var(X)) — automatycznie dobiera do liczby cech.
- `probability=True` — włącza **skalowanie Platt'a**: po treningu SVM dodatkowo trenowana jest logistyczna kalibracja na danych przez 5-fold CV wewnętrzne. Dzięki temu `predict_proba()` zwraca skalibrowane prawdopodobieństwa zamiast surowych wartości funkcji decyzyjnej. Jest wolniejsze od samego SVM, ale niezbędne do rankingu top-5.
- `class_weight='balanced'` — wagi klas odwrotnie proporcjonalne do liczności. Kompensuje nierówną liczbę zdjęć per osoba.

#### Random Forest

**Idea:** Zespół (ensemble) drzew decyzyjnych. Każde drzewo jest trenowane na innym bootstrapowym podzbiorze danych (losowanie ze zwracaniem) i w każdym węźle wybiera najlepszy podział spośród losowego podzbioru `sqrt(n_features)` ≈ 13 cech. Predykcja = głosowanie większościowe wszystkich 200 drzew.

Zalety RF:
- Odporność na przeuczenie (bagging + losowość cech dekoreluje drzewa)
- Brak potrzeby skalowania cech (drzewa są niezależne od skali)
- Naturalnie wieloklasowy
- `feature_importances_` jako bonus (które cechy są najważniejsze)

Parametry:
- `n_estimators=200` — 200 drzew. Więcej = mniejsza wariancja, wolniej. 200 to dobry kompromis dla 382 próbek.
- `max_features='sqrt'` — √181 ≈ 13 losowych cech per podział.
- `class_weight='balanced'` — jak w SVM.
- `n_jobs=-1` — trenowanie równoległe na wszystkich rdzeniach CPU.

#### k-NN — k-Nearest Neighbors

**Idea:** "Leniwy" klasyfikator — nie buduje modelu w fazie treningowej, tylko zapamiętuje wszystkie próbki treningowe. Przy predykcji znajduje k najbliższych (w sensie odległości euklidesowej) próbek treningowych i głosuje.

Parametry:
- `n_neighbors=15` — 15 sąsiadów. Więcej sąsiadów = gładsze granice decyzyjne, mniejsza wariancja.
- `weights='uniform'` — każdy z 15 sąsiadów ma taki sam głos (bez ważenia odległością).
- `metric='euclidean'` — odległość euklidesowa w przestrzeni 181 cech.

**Problem z `predict_proba()`:** Standardowe głosowanie 15 sąsiadów daje dyskretne prawdopodobieństwa (0/15, 1/15, ..., 15/15). Dla biometrii ust, gdzie kilka zdjęć treningowych tej samej osoby ma prawie identyczne wektory cech, `weights='distance'` powoduje że zdjęcie prawie identyczne (odległość ≈ 0) otrzymuje weight ≈ ∞ i daje 100% confidence — kompletnie nieinformatywne.

**Rozwiązanie — `_knn_softmax_proba()`:**
1. Oblicz odległości euklidesowe do **wszystkich** próbek treningowych (nie tylko 15 sąsiadów)
2. Dla każdej klasy weź **minimum** odległości: `d_min[k] = min(dist do próbek klasy k)`
3. Temperatura: `T = median(d_min)` — automatyczne skalowanie
4. Softmax: `p(k) = exp(-d_min[k]/T) / Σ_j exp(-d_min[j]/T)`
5. Stabilizacja numeryczna: odejmij maksimum przed exp

Wynik: rozkład prawdopodobieństwa odzwierciedlający rzeczywistą odległość do każdej klasy. Osoby z podobnymi wektorami cech otrzymują zbliżone (ale nie identyczne) confidence.

---

## 9. Schemat przepływu danych

### Metoda 2 — od zdjęcia do predykcji

```
┌─────────────────────────────────────────────────────┐
│  Zdjęcie PNG (1478×560 px)                          │
└──────────────────────┬──────────────────────────────┘
                       │ LipDataset.load_image(..., color="bgr")
                       ▼
┌─────────────────────────────────────────────────────┐
│  PREPROCESSING  (src/method2_ml/preprocessing.py)   │
│                                                      │
│  BGR  →  Grayscale  →  CLAHE  →  Bilateral  →  Resize│
│         (1 kanał)    (kontrast)  (odszum.)   512×256 │
└──────────────────────┬──────────────────────────────┘
                       │ resized: np.ndarray (256×512, uint8)
                       ▼
┌─────────────────────────────────────────────────────┐
│  EKSTRAKCJA SUROWYCH CECH                            │
│  (src/method2_ml/feature_vector.py)                  │
│                                                      │
│  LBP   → histogram 10-binowy         →  10 wartości │
│  HOG   → gradienty, komórki, bloki   → 16740 wartości│
│  Gabor → 32 filtry × (mean + std)    →  64 wartości │
│  Min.  → szkielet → CN → statystyki  →   7 wartości │
│                                                      │
│  concat → surowy wektor 16821 (float32)              │
│  zapisany do: data/processed/<os>/<stem>/features.npy│
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴──────────┐
              │  FIT (tylko train) │
              ▼                   ▼
┌─────────────────────┐  ┌──────────────────────────┐
│  PCA                │  │  Wymiary znane            │
│  HOG: 16740 → 100   │  │  lbp_dim=10, hog_dim=100 │
│  fit na X_train_hog │  │  gabor_dim=64, min_dim=7  │
└──────────┬──────────┘  └──────────────────────────┘
           │ transform (train + test)
           ▼
┌─────────────────────────────────────────────────────┐
│  PO PCA: wektor 181 cech (10+100+64+7)              │
└──────────────────────┬──────────────────────────────┘
                       │
              ┌────────┴──────────┐
              │  FIT (tylko train) │
              ▼
┌─────────────────────────────────────────────────────┐
│  StandardScaler                                      │
│  mean=0, std=1 per cecha                             │
│  fit na X_train_181                                  │
└──────────────────────┬──────────────────────────────┘
                       │ zestandaryzowany X (n × 181)
                       ▼
┌────────────┐  ┌────────────┐  ┌────────────┐
│    SVM     │  │     RF     │  │    k-NN    │
│  rbf, C=10 │  │  200 drzew │  │  15 sąs.  │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
                      ▼
       Top-5 kandydatów + confidence per klasa
       + konsensus 3 klasyfikatorów
```

### Cross-validation (5-fold)

```
382 zdjęcia (22 osoby)
       │
       ▼  StratifiedKFold(n=5, shuffle=True, seed=42)
       │
       ├─ Fold 0: train ~306 zdjęć (80%), test ~76 (20%)
       ├─ Fold 1: train ~306, test ~76   (inne zdjęcia testowe)
       ├─ Fold 2: train ~306, test ~76
       ├─ Fold 3: train ~306, test ~76
       └─ Fold 4: train ~306, test ~76   (każda osoba ~3-4 razy w teście)

Zapisane: data/splits/fold_0.json ... fold_4.json (reprodukowalne)
Metryka końcowa: mean ± std z 5 przebiegów
```

---

## 10. Aplikacja Streamlit — opis każdej strony

### Sidebar (widoczny na każdej stronie)

Zaimplementowany w `apps/sidebar_shortcuts.py`. Zawiera szybkie skróty do folderów i plików projektu — kliknięcie przycisku otwiera lokalizację w Eksploratorze Windows (`os.startfile()`).

Przyciski:
- **📸 Baza danych (N zdjęć)** — otwiera `data/baza_danych/`
- **🤖 Modele (N plików)** — otwiera `models/`
- **📊 Excel predykcji (N wierszy)** — otwiera `exports/predykcje_szczegolowe.xlsx` (liczy wiersze w arkuszu Predykcja przez openpyxl w trybie read_only)
- **📋 Ostatnia ewaluacja** — otwiera najnowszy plik `.xlsx` z `results/`
- **🖼️ Wykresy (N plików)** — otwiera `results/plots/`
- **💾 Folder eksportów (N plików)** — otwiera `exports/`
- **(Rozwijany) 🔧 Dane pośrednie**: cache `.npy`, pliki JSON splits, katalog główny

---

### Metoda 2: Klasyczne ML

#### Strona 1 — Eksploracja danych (`page_01_eksploracja.py`)

Cel: przegląd i zrozumienie bazy danych przed jakimkolwiek przetwarzaniem.

Zawartość:
- Metryki zbiorcze: liczba osób, łączna liczba zdjęć, zdjęcia/osobę min/max/średnia
- Wykres słupkowy liczby zdjęć per osoba (identyfikacja nierównomierności)
- Rozkład lampa ON vs OFF (histogram)
- Rozkład parametrów technicznych: fokus, ekspozycja
- Sekcja galerii: wybór osoby z dropdown → siatka wszystkich jej zdjęć z `st.image()`
- Karta metadanych wybranej osoby z pliku xlsm (płeć, wiek, skóra, cechy szczególne)

Dataset ładowany przez `@st.cache_resource` — wczytany raz, zachowany między nawigacjami.

---

#### Strona 2 — Preprocessing (`page_02_preprocessing.py`)

Cel: wizualizacja każdego kroku pipeline preprocessingu z możliwością zmiany parametrów.

Zawartość:
- Wybór zdjęcia: dropdown (wszystkie PNG z bazy) lub przycisk "Losuj"
- Suwaki parametrów: `clip_limit` CLAHE (0.5–8.0), `sigma_color/space` bilateral (10–150)
- Wyświetlenie 5 kroków obok siebie: Original → Grayscale → CLAHE → Bilateral → Resize
- Statystyki jasności na każdym etapie: mean, std, min, max
- Przycisk "Eksportuj do dokumentacji" — kopiuje obrazy pośrednie do `exports/<timestamp>/` z opisowymi nazwami plików (do screenshotów do pracy)

---

#### Strona 3 — Ekstrakcja cech (`page_03_features.py`)

Cel: wizualizacja co "widzi" każdy ekstraktor cech dla wybranego zdjęcia.

Zawartość:
- **LBP**: heatmapa wzorców LBP nałożona na obraz + histogram 10-binowy
- **HOG**: wizualizacja gradientów (`hog(..., visualize=True)`) nałożona na obraz
- **Gabor**: siatka 4×8 odpowiedzi wszystkich filtrów (wiersze = częstotliwości, kolumny = orientacje)
- **Minucje**: oryginalny obraz z nałożonym szkieletem bruzd i kolorowymi kropkami minucji (🟢 ending, 🔴 bifurcation, 🔵 crossing)
- Tabela wartości cech globalnych: 7 wartości minucji, 10 binów LBP, energia HOG, orientacja Gabora

---

#### Strona 4 — Trening modelu (`page_04_training.py`)

Cel: trenowanie klasyfikatorów z pełnym podglądem procesu.

**Status datasetu** (sekcja u góry):
- Oblicza aktualny hash SHA-256 bazy danych
- Porównuje z `models/dataset_hash.txt`
- Wyświetla: "✅ Model aktualny" lub "⚠️ Wykryto zmiany — zalecany retrain"

**Ustawienia:**
- Radio: SVM / RF / k-NN / wszystkie 3
- Radio: 1 fold (szybki podgląd) lub 5-fold CV (pełna ewaluacja)
- Suwak PCA HOG: 20–200 składowych (domyślnie 100)
- Checkbox "Używaj cache wektorów cech" (domyślnie True)

**Przebieg treningu** (po kliknięciu przycisku):
1. Pasek postępu per etap (ekstrakcja train, ekstrakcja test, trening)
2. Log tekstowy z metrykami każdego folda
3. Po zakończeniu: tabela per fold + agregaty mean ± std
4. Zapis modelu: `models/method2_<clf>.joblib`
5. Zapis pipeline'u: `models/method2_pipeline.joblib`
6. Zapis nowego hash datasetu

Uwaga: model jest zapisywany z **ostatniego folda** (nie z uśrednionego) — przy predykcji używamy jednego konkretnego modelu, nie ensemble z 5 foldów.

---

#### Strona 5 — Predykcja (`page_05_prediction.py`)

Kluczowa strona — identyfikacja dowolnego zdjęcia ust.

**Wybór zdjęcia** (3 zakładki):
1. **Losowe z bazy** — przycisk "Wylosuj" → losuje z 382 zdjęć
2. **Ręczny wybór** — dropdown z wszystkimi zdjęciami (filtrowanie po numerze osoby)
3. **Upload** — przesłanie zewnętrznego pliku PNG

**Analiza** (po kliknięciu "Analizuj"):
1. Preprocessing — 5 miniaturek kroków pipeline
2. Ekstrakcja cech (z cache lub od nowa)
3. Załadowanie modeli i pipeline'u z `models/`
4. Predykcja **wszystkimi 3 klasyfikatorami jednocześnie** (SVM, RF, k-NN)

**Prezentacja wyników** — 3 kolumny (jedna per klasyfikator):
- Top-1: nazwa osoby (Nr_XX) + zdjęcie referencyjne tej osoby z bazy
- Confidence top-1 (procent)
- Karta metadanych top-1 (płeć, wiek, skóra)
- Wykres słupkowy top-5 kandydatów:
  - 🟩 zielony = poprawna odpowiedź
  - 🟦 niebieski = predykcja top-1 (jeśli niepoprawna)
  - ⬜ szary = pozostałe
- ✅ / ❌ znacznik poprawności (tylko dla zdjęć z bazy, gdzie znana jest etykieta)

**Sekcja Konsensus** (pod kolumnami):
- "Wszystkie 3 klasyfikatory wskazały: Nr_XX" (gdy zgodne)
- "2/3 wskazały Nr_XX, RF wskazało Nr_YY" (gdy niezgodne)

**Sekcje rozwijane** (`st.expander`):
- Szczegóły preprocessingu (parametry, statystyki jasności)
- Cechy minucji (tabela 7 wartości)
- Metadane top-3 kandydatów każdego klasyfikatora

**Zapis do Excela** — przycisk "Zapisz" → dopisuje 78-kolumnowy wiersz do `exports/predykcje_szczegolowe.xlsx`.

---

#### Strona 6 — Ewaluacja (`page_06_evaluation.py`)

Cel: pełna naukowa ewaluacja klasyfikatorów przez 5-fold CV.

**Konfiguracja:**
- Radio: SVM / RF / k-NN / wszystkie 3
- Suwak PCA HOG (musi być zgodne z trenowaniem)
- Checkbox cache

**Uruchomienie** (przycisk "Uruchom ewaluację"):
- Dla każdego folda × każdego klasyfikatora:
  - Fit pipeline (PCA + scaler) na train
  - Transform test
  - **Trening od zera** (fresh model per fold, nie ładuje z dysku) — ważne: każdy fold ma własny, niezależny model dla rzetelnego pomiaru
  - Obliczenie: accuracy, top-3, top-5, precision (macro), recall (macro), F1 (macro), czas

**Wyniki per klasyfikator:**
- Tabela: fold × metryki
- Agregaty: mean ± std dla accuracy, top-5, F1
- Macierz pomyłek (heatmapa seaborn, zsumowana z 5 foldów): oś X = predykowana klasa, oś Y = prawdziwa klasa
- Auto-zapis wykresu do `results/plots/confusion_matrix_<clf>.png`

**Porównanie** (gdy "wszystkie 3"): tabela porównawcza accuracy/top-5/F1 per klasyfikator.

**Eksport** — przycisk "Eksportuj do Excela" → zapisuje `results/experiment_<timestamp>.xlsx`.

---

### Metoda 1: Tradycyjna

#### Strona 7 — Predykcja obraz–obraz (`page_prediction.py`)

Identyczna struktura wyboru zdjęcia jak Strona 5 (3 zakładki: losowe/ręczne/upload).

Wybór miary: radio SSIM / ORB / Histogram / Combined.

**Przebieg:**
1. Ładowanie całego zbioru treningowego folda 0 do pamięci
2. Obliczenie score do każdego z ~306 zdjęć treningowych (pasek postępu)
3. Sortowanie malejąco po score

**Wyniki:**
- Top-5 kandydatów: miniatura + numer osoby + score + metadane
- Wyróżnienie poprawnej odpowiedzi (jeśli zdjęcie z bazy)
- Czas obliczenia w sekundach

---

#### Strona 8 — Ewaluacja CV Metody 1 (`page_evaluation.py`)

5-fold CV dla miar podobieństwa.

- Wybór miary (lub wszystkie 4)
- Pasek postępu per fold per zdjęcie testowe
- Tabela accuracy/top-3/top-5 per fold
- Macierz pomyłek
- Porównanie miar (gdy wybrano >1)
- Eksport do Excela

---

## 11. Pliki wyników i format Excela

### `results/experiment_<timestamp>.xlsx`

Generowany przez Stronę 6 (ewaluacja). Format timestamp: `experiment_2026-04-13_14-30-00.xlsx`.

Arkusze:

**`Run_info`** — jeden arkusz z parami klucz-wartość:
- `timestamp`, `HOG_PCA_components`, `Klasyfikatory`, `N_folds`

**`Predictions`** — jeden wiersz per predykcja per fold:
```
image_path | true_id | method | predicted_id | top3 | top5 | confidence | time_ms | correct
```

**`Summary_svm`** (i analogiczne dla rf, knn) — jeden wiersz per fold + wiersz MEAN i STD:
```
fold | accuracy | top3_accuracy | top5_accuracy | precision | recall | f1 | time_s
```

**`ConfusionMatrix_svm`** (i analogiczne) — macierz 22×22 z etykietami klas jako nagłówkami wierszy/kolumn.

---

### `results/plots/confusion_matrix_<clf>.png`

Heatmapa seaborn (DPI=150, bbox_inches='tight'). Etykiety osi: `Nr_01` ... `Nr_22`. Format: PNG gotowy do wklejenia do pracy magisterskiej. Generowany automatycznie przy każdej ewaluacji.

---

### `exports/predykcje_szczegolowe.xlsx`

Szczegółowy log predykcji z GUI (Strona 5). Każde kliknięcie "Zapisz do Excela" dopisuje jeden wiersz. Plik zawiera:

- Arkusz **`Predykcja`** — 78 kolumn, nagłówki zamrożone (frozen header), auto-dopasowane szerokości kolumn
- Arkusz **`Legenda`** — kolorowy słownik wszystkich 78 kolumn z opisem i jednostką

---

## 12. Konwencje nazewnictwa zdjęć

### Format nazwy pliku

```
2026-03-05_08-25-34_Nr_2_LampaOFF_Fokus_0_68_Exp_0_CUT.png
│            │       │       │         │       │    │
│            │       │       │         │       │    └─ CUT = zdjęcie wycięte do obszaru ust
│            │       │       │         │       └────── Ekspozycja (liczba całkowita, może być ujemna)
│            │       │       │         └────────────── Fokus (podkreślnik zamiast kropki: 0_68 = 0.68)
│            │       │       └──────────────────────── LampaON lub LampaOFF
│            │       └──────────────────────────────── Nr_X lub Nr_XX (z/bez zera wiodącego)
│            └──────────────────────────────────────── Godzina: HH-MM-SS
└───────────────────────────────────────────────────── Data: YYYY-MM-DD
```

### Dodawanie nowych zdjęć

1. Skopiuj pliki PNG do `data/baza_danych/` (zgodnie z formatem nazwy)
2. Otwórz aplikację → Strona 4 (Trening)
3. Pojawi się ostrzeżenie "⚠️ Wykryto zmiany w bazie danych"
4. Kliknij "Trenuj" — modele zostaną wytrenowane na nowym zbiorze
5. Pliki JSON z podziałem foldów zostaną automatycznie zregenerowane (stare są nieaktualne)
6. Cache `.npy` dla starych zdjęć pozostaje ważny — tylko nowe zdjęcia zostaną przeliczone

---

## 13. Mechanizm auto-retrain (hash datasetu)

### Problem

Jeśli użytkownik doda nowe zdjęcia do bazy, zapisane modele stają się nieaktualne (nie "widziały" nowych danych). Bez wykrycia tej zmiany predykcje byłyby na modelach trenowanych na innym zbiorze niż dostępne dane.

### Rozwiązanie — SHA-256 hash

1. Przy wejściu na Stronę 4 obliczany jest `compute_dataset_hash()` = SHA-256 z `(nazwa+rozmiar+mtime)` każdego pliku w `data/baza_danych/`
2. Hash porównywany z `models/dataset_hash.txt` (zapisanym po ostatnim trenowaniu)
3. Jeśli różny lub plik nie istnieje → wyświetlane ostrzeżenie, przycisk "Trenuj" podświetlony
4. Po zakończeniu trenowania nowy hash zapisywany

**Nie trenujemy automatycznie** — użytkownik musi kliknąć przycisk. Daje to pełną kontrolę i możliwość udokumentowania momentu trenowania.

---

## 14. Typowe czasy działania

Czasy mierzone orientacyjnie na laptopie z Intel Core i5/i7, 16 GB RAM, bez GPU.

| Operacja | Czas | Uwagi |
|---|---|---|
| Wczytanie metadanych xlsm | ~2 s | openpyxl, 22 arkusze |
| Ekstrakcja cech — 1 zdjęcie (bez cache) | ~0.15 s | LBP+HOG+Gabor+Minutiae |
| Ekstrakcja cech — 1 zdjęcie (z cache .npy) | ~0.005 s | np.load() |
| Ekstrakcja całej bazy 382 zdjęć (bez cache) | ~60–90 s | dominuje Gabor |
| Ekstrakcja całej bazy (z cache) | ~1–2 s | tylko np.load() |
| Fit PCA (HOG 16740→100) | ~2–5 s | SVD na macierzy 306×16740 |
| Trening SVM (C=10, rbf) + Platt scaling | ~5–15 s | OvO, 231 klasyfikatorów |
| Trening RF (200 drzew, n_jobs=-1) | ~10–30 s | zależy od liczby rdzeni |
| Trening k-NN (indeksowanie) | <1 s | lazy — tylko zapisuje dane |
| 1 fold (1 klasyfikator) | ~3–10 min | przy pierwszym uruchomieniu bez cache |
| 5-fold CV (1 klasyfikator) | ~10–25 min | z cache: ~2–5 min |
| 5-fold CV (wszystkie 3 klasyfikatory) | ~30–75 min | z cache: ~5–15 min |
| Predykcja 1 zdjęcia (z cache, model załadowany) | <1 s | |
| Predykcja 1 zdjęcia (bez cache) | ~1–2 s | ekstrakcja + predykcja |
| Metoda 1 — SSIM dla 1 zdjęcia testowego | ~5–15 s | 306 porównań SSIM 256×128 |
| Metoda 1 — ORB dla 1 zdjęcia testowego | ~15–30 s | 306 dopasowań keypointów |
| Metoda 1 — Histogram dla 1 zdjęcia | ~1–3 s | najszybsza miara |

---

## 15. Konfiguracja projektu

### `.streamlit/config.toml`

```toml
[server]
maxUploadSize = 50   # maks. rozmiar pliku przesyłanego przez upload (MB)
port = 8502          # port HTTP (zmieniony z 8501 aby uniknąć konfliktów)

[theme]
primaryColor          = "#2E75B6"  # niebieski — akcenty, przyciski
backgroundColor       = "#FFFFFF"  # białe tło
secondaryBackgroundColor = "#F0F2F6"  # jasnoszare tło sidebarów
textColor             = "#1A1A2E"  # ciemnoniebieska czerń
font                  = "sans serif"
```

### `src/core/config.py`

Wszystkie stałe projektu w jednym miejscu. Zmiana `TARGET_SIZE` wymaga usunięcia cache `.npy` i ponownego trenowania (bo zmienia wymiar wektora cech). Zmiana `RANDOM_SEED` zmienia podziały foldów (ale tylko jeśli usuniesz pliki JSON — inaczej stare są wczytywane).

### Reprodukowalność wyników

- `RANDOM_SEED = 42` używany we wszystkich miejscach losowości: `StratifiedKFold`, `SVC`, `RandomForestClassifier`, `PCA`
- Foldy CV zapisane do JSON → identyczne między sesjami
- Wektory cech cache'owane w `.npy` → deterministyczna ekstrakcja
- `weights='uniform'` w k-NN eliminuje zależność od kolejności próbek (problem przy `weights='distance'` i distance=0)

---

## 16. Biblioteki i technologie — szczegółowy opis

### Streamlit

**Co to jest?** Streamlit to framework open-source do tworzenia aplikacji webowych w czystym Pythonie. Nie wymaga znajomości HTML/CSS/JavaScript — cała aplikacja pisana jest jako skrypt Python. Streamlit uruchamia lokalny serwer HTTP (Tornado) i serwuje strony do przeglądarki.

**Jak działa model wykonania?** Streamlit ma **reaktywny model wykonania** (ang. reactive execution model): za każdym razem gdy użytkownik zmieni widget (suwak, przycisk, dropdown), **cały skrypt jest wykonywany od nowa** od góry do dołu. To prostsze w implementacji niż tradycyjne frameworki (Flask, Django), ale wymaga ostrożności z operacjami ciężkimi obliczeniowo.

**Cache:** `@st.cache_resource` cache'uje wynik funkcji przez całe życie serwera (między nawigacjami stron i przeładowaniami). Używane do cachowania `LipDataset` i metadanych — wczytane raz przy pierwszej wizycie, potem dostępne natychmiast.

**Wielostronicowość:** projekt używa `st.navigation()` z `st.Page()` — nowszy mechanizm (≥1.30) pozwalający na grupowanie stron w sekcje i definiowanie ich jako funkcji Pythona (nie osobnych plików).

**Wersja projektu:** ≥1.30 (wymagane dla `st.navigation`).

---

### OpenCV (`cv2`)

**Co to jest?** OpenCV (Open Source Computer Vision Library) to biblioteka napisana w C++ z bindingami dla Pythona, zawierająca ponad 2500 zoptymalizowanych algorytmów przetwarzania obrazu i widzenia komputerowego. Jest standardem de facto w dziedzinie.

**Używane algorytmy w projekcie:**

**CLAHE** (`cv2.createCLAHE`): Contrast Limited Adaptive Histogram Equalization. Wersja lokalna standardowego HE (Histogram Equalization) — zamiast wyrównywać histogram całego obrazu globalnie (co może prześwietlać jasne obszary), dzieli obraz na siatkę małych kafelków i wyrównuje każdy osobno. "Contrast Limited" = clip limit ogranicza maksymalną amplifikację w histogramie, co zapobiega wzmacnianiu szumu.

**Bilateral Filter** (`cv2.bilateralFilter`): Filtr który usuwa szum zachowując ostre krawędzie. Standardowe filtry rozmywające (Gaussian, median) traktują wszystkich sąsiadów jednakowo. Bilateral filter waży sąsiadów podwójnie: im dalej przestrzennie (sigma_space) i im bardziej różni kolorystycznie (sigma_color), tym mniejszy wpływ. Piksele po drugiej stronie krawędzi (inny kolor) mają minimalny wpływ na środkowy piksel.

**Adaptive Threshold** (`cv2.adaptiveThreshold`): Binaryzacja z lokalnie obliczanym progiem — w każdym bloku pikseli osobno. Potrzebne gdy oświetlenie jest nierównomierne (jedna część obrazu jaśniejsza, inna ciemniejsza).

**ORB** (`cv2.ORB_create`): szybki detektor punktów kluczowych oparty na FAST + BRIEF, bez licencji patentowych.

**`cv2.getGaborKernel`**: tworzenie jąder filtrów Gabora z zadanymi parametrami.

---

### scikit-image

**Co to jest?** scikit-image to biblioteka przetwarzania obrazu zbudowana na NumPy/SciPy. Uzupełnia OpenCV o algorytmy bardziej matematyczne i akademickie.

**Używane w projekcie:**

**`local_binary_pattern`** (skimage.feature): implementacja LBP z opcją 'uniform' (wzorce z ≤2 przejściami 0↔1).

**`hog`** (skimage.feature): implementacja HOG z normalizacją L2-Hys bloków i opcją wizualizacji gradientów (parametr `visualize=True`).

**`structural_similarity`** (skimage.metrics): implementacja SSIM według oryginalnej publikacji Wang et al. (2004). Zwraca zarówno globalny score jak i mapę lokalnego SSIM (dla wizualizacji).

**`skeletonize`** (skimage.morphology): algorytm skeletonizacji morfologicznej — iteracyjne usuwanie pikseli z brzegów obiektów binarnych przy zachowaniu połączalności topologicznej. Redukuje obiekty 2D do ich "kośćca" o szerokości 1 piksel.

---

### scikit-learn

**Co to jest?** scikit-learn to wiodąca biblioteka Machine Learning dla Pythona. Cechuje się spójnym API: każdy model ma `fit()`, `predict()`, `transform()` / `fit_transform()`. Projekt używa jej intensywnie.

**`SVC`** (Support Vector Classifier): implementacja SVM z optymalizacją LIBSVM. Dla wieloklasowego problemu (22 klasy) używa strategii OvO (One-vs-One) — trenuje C(22,2) = 231 binarnych SVM. Przy `probability=True` dodaje kalibrację Platt'a przez wewnętrzne 5-fold CV.

**`RandomForestClassifier`**: implementacja RF. Każde drzewo trenowane na bootstrapowym podzbiorze danych i losowym podzbiorze cech per węzeł. `n_jobs=-1` = używa wszystkich rdzeni CPU.

**`KNeighborsClassifier`**: standardowy k-NN z wyszukiwaniem KD-Tree lub Ball-Tree dla szybkiego znajdowania sąsiadów.

**`PCA`** (Principal Component Analysis): redukcja wymiarowości przez SVD (Singular Value Decomposition). Przy `n_components=100` zachowuje 100 kierunków o największej wariancji. Dla HOG (16740 cech) redukuje do 100 niezależnych (ortogonalnych) składowych.

**`StandardScaler`**: normalizacja z-score: `x_scaled = (x - mean) / std`. `fit()` oblicza mean i std na danych treningowych, `transform()` stosuje tę samą transformację do danych testowych.

**`StratifiedKFold`**: podział zachowujący proporcje klas w każdym foldzie. Dla 22 klas i 382 próbek zapewnia że każda osoba ma ~3–4 zdjęcia w każdym zbiorze testowym.

**`pairwise_distances`**: obliczenie macierzy odległości między dwoma zbiorami próbek. Używane w `_knn_softmax_proba()` do obliczenia odległości do wszystkich próbek treningowych naraz.

**`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`**: standardowe metryki klasyfikacji. `average='macro'` = średnia nieważona po wszystkich klasach (każda klasa ma równy udział niezależnie od liczności).

---

### NumPy

**Co to jest?** NumPy (Numerical Python) to fundamentalna biblioteka do obliczeń numerycznych w Pythonie. Dostarcza typ `ndarray` — wielowymiarowa tablica z szybkimi operacjami wektoryzowanymi (implementacja w C/Fortran).

**Używane w projekcie:**
- Macierze cech: `X_train` (306×181), `X_test` (76×181)
- Wektory etykiet: `y_train`, `y_test`
- Obliczenia softmax w `_knn_softmax_proba()`
- `np.argsort(proba)[::-1][:5]` — ranking top-5 kandydatów
- `np.save()` / `np.load()` — cache wektorów cech
- `np.concatenate()` — sklejenie wektorów LBP+HOG+Gabor+Minutiae
- `np.mean()`, `np.std()` — agregacja metryk CV

---

### pandas

**Co to jest?** pandas to biblioteka analizy danych. Dostarcza `DataFrame` — tabelę z etykietowanymi kolumnami i wierszami, inspirowaną R.

**Używane w projekcie:**
- Wyświetlanie tabel metryk w GUI przez `st.dataframe(pd.DataFrame(...))`
- Konwersja wyników per fold do tabeli (accuracy, top3, top5, F1, czas)
- Tabela cech globalnych na Stronie 3

---

### openpyxl

**Co to jest?** openpyxl to czysto pythonowa biblioteka do odczytu i zapisu plików Excel (.xlsx, .xlsm). Nie wymaga zainstalowanego programu Microsoft Excel.

**Używane w projekcie:**

*Odczyt metadanych:* `load_workbook(xlsm_path, data_only=True, keep_vba=False)`. `data_only=True` = zwraca obliczone wartości komórek (nie formuły). `keep_vba=False` = ignoruje makra VBA (przyspiesza).

*Zapis wyników ewaluacji (`ExperimentResults.save()`):* tworzy wieloarkuszowy `.xlsx` z formatowaniem:
- Pogrubione nagłówki kolumn z kolorowym wypełnieniem
- Zamrożone wiersze nagłówkowe (`ws.freeze_panes`)
- Auto-dopasowanie szerokości kolumn przez iterację po wartościach
- Wyrównanie tekstów (`Alignment`)

*Zapis szczegółowych predykcji (`write_detailed_prediction_excel()`):`* dopisywanie wierszy do istniejącego pliku (`load_workbook` + `append` + `save`). Arkusz Legenda tworzony tylko przy pierwszym wywołaniu.

*Odczyt liczby wierszy do widgetu sidebar:* `load_workbook(..., read_only=True)` + `ws.max_row` — szybki odczyt bez parsowania całego pliku.

---

### matplotlib

**Co to jest?** matplotlib to podstawowa biblioteka do tworzenia wykresów 2D w Pythonie. Oferuje niski poziom kontroli (każdy element wykresu konfigurowalny). API inspirowane MATLABem.

**Używane w projekcie:**
- Rysowanie macierzy pomyłek jako heatmap (`fig, ax = plt.subplots()`)
- Zapisywanie wykresów do PNG: `fig.savefig(path, dpi=150, bbox_inches='tight')`
- Wizualizacja histogramów cech na Stronie 3
- Siatka odpowiedzi filtrów Gabora

---

### seaborn

**Co to jest?** seaborn to biblioteka wizualizacji statystycznej zbudowana na matplotlib. Oferuje wyższy poziom abstrakcji i piękniejsze domyślne style.

**Używane w projekcie:** wyłącznie `sns.heatmap()` do rysowania macierzy pomyłek:

```python
sns.heatmap(
    cm_total,                   # macierz 22×22
    annot=True,                  # liczby w komórkach
    fmt="d",                     # format integer
    cmap="Blues",                # skala niebieska
    xticklabels=classes_labels,  # etykiety osi X
    yticklabels=classes_labels,  # etykiety osi Y
    linewidths=0.3               # cienkie linie siatki
)
```

---

### joblib

**Co to jest?** joblib to biblioteka do serializacji obiektów Python i równoległego przetwarzania. Używana przez scikit-learn wewnętrznie do zapisywania modeli i równoległego trenowania RF.

**Używane w projekcie:** zapis i odczyt wytrenowanych modeli:

```python
joblib.dump(clf_object, "models/method2_svm.joblib")  # zapis
clf = joblib.load("models/method2_svm.joblib")        # odczyt
```

`joblib.dump` jest szybszy niż standardowe `pickle` dla obiektów zawierających duże tablice NumPy (używa memmap dla wydajności). Plik `.joblib` zawiera pełny stan obiektu Python — po wczytaniu jest gotowy do `predict()` bez ponownego trenowania.

---

### hashlib (biblioteka standardowa Python)

**Co to jest?** Moduł standardowej biblioteki Pythona do obliczania skrótów kryptograficznych (SHA-1, SHA-256, MD5 itp.).

**Używane w projekcie:** obliczanie SHA-256 hash stanu folderu z danymi:

```python
combined = "\n".join(entries).encode("utf-8")
hashlib.sha256(combined).hexdigest()  # → 64-znakowy hex string
```

SHA-256 produkuje 256-bitowy (32-bajtowy) skrót — prawdopodobieństwo kolizji dla plików o różnej treści jest astronomicznie małe (2⁻²⁵⁶).

---

### pathlib (biblioteka standardowa Python)

**Co to jest?** Moduł standardowej biblioteki Python do obiektowego zarządzania ścieżkami systemu plików. Zastępuje przestarzałe `os.path`.

**Używane w projekcie wszędzie:** ścieżki jako `Path` obiekty zamiast stringów:

```python
path = MODELS_DIR / "method2_svm.joblib"  # łączenie ścieżek operatorem /
path.exists()           # sprawdzenie istnienia
path.parent.mkdir(parents=True, exist_ok=True)  # tworzenie folderów
path.stem               # nazwa bez rozszerzenia
path.stat().st_mtime    # czas modyfikacji (do cache invalidation)
sorted(folder.glob("*.png"))  # listowanie plików
```

---

### albumentations

**Co to jest?** albumentations to szybka biblioteka do augmentacji danych obrazowych, zoptymalizowana pod uczenie maszynowe. Obsługuje ponad 70 transformacji.

**Status w projekcie: zainstalowana ale NIEUŻYWANA.** Planowano augmentację treningową (rotacja ±5°, zmiana jasności/kontrastu ±15%, flip horyzontalny), ale zrezygnowano z niej w finalnej wersji projektu. Augmentacja mogła poprawić accuracy przy tak małym zbiorze danych (382 zdjęć), ale komplikuje reprodukowalność i interpretację wyników. Biblioteka pozostaje w `requirements.txt` jako opcjonalna.

---

## 17. Ograniczenia projektu

Projekt świadomie **nie obejmuje** następujących zagadnień:

| Temat | Powód pominięcia |
|---|---|
| Deep learning (CNN, transfer learning, Siamese) | Praca skupia się na interpretowalności — CNN jest "czarną skrzynką", cechy LBP/HOG/Gabor/minucje są czytelne i opisywalne |
| Weryfikacja 1-do-1 | Tylko identyfikacja 1-do-N; weryfikacja wymaga innego podejścia (Siamese, cosine similarity) |
| Podział ust na kwadranty/sektory | Całe usta jako jedna strefa — uproszczenie uzasadnione małym zbiorem |
| Klasyfikacja typów bruzd Suzuki-Tsuchihashi | Możliwe rozszerzenie, poza zakresem pracy |
| Detekcja ust w obrazie (YOLO, dlib) | Wszystkie zdjęcia już wycięte (`_CUT`) |
| Uczenie przyrostowe | Nowe zdjęcia = pełny retrain |
| Cross-platform skróty | `os.startfile()` — tylko Windows |
| Augmentacja danych | Usunięta — komplikuje reprodukowalność |
| GPU acceleration | Zbędne dla 382 próbek i 181 cech |
| Weryfikacja jakości zdjęć | Brak filtrowania rozmazanych/źle wykadrowanych zdjęć |

---

## 18. Rozwiązywanie problemów

### `FileNotFoundError: data/baza_danych/`
Folder z bazą danych nie istnieje lub jest pusty. Sprawdź ścieżkę w `src/core/config.py`. Upewnij się że uruchamiasz z katalogu projektu (`cd "c:/Magisterka/15 projekcik"`).

### `ModuleNotFoundError: No module named 'cv2'`
```bash
pip install opencv-python
```

### `ModuleNotFoundError: No module named 'streamlit'`
```bash
pip install -r requirements.txt
```

### Streamlit nie uruchamia się — port 8502 zajęty
Sprawdź czy poprzednia instancja nadal działa:
```bash
netstat -ano | findstr :8502
```
Zmień port w `.streamlit/config.toml` lub zamknij poprzednią instancję.

### Strona 5 — "Brak wytrenowanego modelu"
Najpierw przejdź na Stronę 4 i wytrenuj co najmniej jeden klasyfikator. Pliki `.joblib` muszą istnieć w `models/`.

### Cache cech — stare/nieprawidłowe wyniki
Zmiana parametrów preprocessingu lub ekstrakcji powoduje że stary cache `.npy` jest nieaktualny. Na Stronie 4 odznacz "Używaj cache wektorów cech" i przelicz od nowa. Można też ręcznie usunąć folder `data/processed/`.

### k-NN — pewność 100% dla wszystkich predykcji
Problem historyczny — rozwiązany przez `_knn_softmax_proba()`. Jeśli nadal występuje: sprawdź czy plik `models/method2_knn.joblib` jest aktualny (nie z poprzedniej wersji kodu). Usuń plik i wytrenuj ponownie.

### `PermissionError` przy zapisie modelu / Excela
Zamknij otwarty plik `.xlsx` lub `.joblib` w Microsoft Excel / Eksploratorze Windows przed ponownym zapisem aplikacji.

### `StreamlitAPIException: Multiple Pages with URL pathname 'render'`
Każda strona musi mieć unikalny `url_path` w `st.Page()`. Sprawdź `apps/app.py` — wszystkie strony mają zdefiniowane URL: `eksploracja`, `preprocessing`, `features`, `training`, `prediction`, `evaluation`, `m1_prediction`, `m1_evaluation`.

### Bardzo wolna pierwsza ekstrakcja cech
Normalne przy pierwszym uruchomieniu — Gabor dla 382 zdjęć to ~60–90 s. Przy kolejnych uruchomieniach z włączonym cache czas spada do ~2 s.

### Skróty w sidebarze nie działają
Funkcja `os.startfile()` jest dostępna tylko na Windows. Na Linux/Mac skróty są wyłączone (disabled), reszta aplikacji działa normalnie.

---

## 19. Różnice względem pierwotnych notatek projektowych

Plik `notatki/01_początek.md` zawiera pierwotne założenia projektowe. Poniżej najważniejsze rozbieżności między planem a finalną implementacją:

| Temat | Notatki — plan | Aktualny stan |
|---|---|---|
| Punkt wejścia aplikacji | Dwie osobne: `app_method1.py` i `app_method2.py` | Jedna ujednolicona: `apps/app.py` (8 stron, 2 sekcje) |
| Port Streamlit | 8501 (domyślny) | **8502** — skonfigurowane w `.streamlit/config.toml` |
| k-NN — liczba sąsiadów | `n_neighbors=5` | **`n_neighbors=15`** — więcej sąsiadów = gładsze granice |
| k-NN — wagi | `weights='distance'` | **`weights='uniform'`** — dystansowe dawały 100% dla bliskich zdjęć |
| k-NN — metoda prawdopodobieństwa | Standardowe głosowanie | **Własny softmax z minimalnych odległości** (`_knn_softmax_proba`) |
| Predykcja (Strona 5) | Radio do wyboru klasyfikatora | **Zawsze wszystkie 3 jednocześnie** — 3 kolumny wyników |
| Excel predykcji | Arkusz `Predictions` w wynikach CV | **Osobny plik `predykcje_szczegolowe.xlsx` (78 kolumn + Legenda)** |
| `augmentation.py` | Moduł albumentations (rotacja, jasność, flip) | **Usunięty** — augmentacja nie jest stosowana w finalnym pipeline |
| `visualization.py` | Planowany helper w `src/core/` | **Nie zaimplementowany** — wizualizacja inline w stronach |
| `FeatureStats` (Excel) | Arkusz ze statystykami cech per osoba | **Usunięty** — informacja dostępna w predykcje_szczegolowe.xlsx |
| `MetadataAnalysis` (Excel) | Arkusz z accuracy per osoba | **Nie zaimplementowany** — możliwa analiza post-hoc z Excela predykcji |
| Sidebar | Nieplanowany | **`sidebar_shortcuts.py`** — skróty do folderów i plików |
| Struktura `apps/` | Płaska (tylko app_method1.py i app_method2.py) | **Hierarchiczna** — `method1_pages/` i `method2_pages/` jako podpakiety |

---

## 20. Słownik pojęć — pełne nazwy, tłumaczenia, wyjaśnienia

Wszystkie skróty używane w projekcie wraz z pełną nazwą angielską, polskim tłumaczeniem i krótkim wyjaśnieniem.

---

### Metody ekstrakcji cech

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **LBP** | Local Binary Patterns | Lokalne wzorce binarne | Teksturowy deskryptor obrazu: dla każdego piksela porównuje jego wartość z 8 sąsiadami (P=8, R=1). Wynik to 8-bitowy kod binarny — np. `11001010`. Kody są grupowane w histogram 10-binowy (metoda `uniform`). Dobry do opisu faktur i bruzd. |
| **HOG** | Histogram of Oriented Gradients | Histogram orientacji gradientów | Deskryptor kształtu: obraz jest dzielony na komórki 16×16 px, w każdej obliczany jest histogram 9 kierunków gradientów (krawędzi). Komórki grupowane w bloki 2×2 i normalizowane. Surowy wektor ma 16 740 wartości, redukowany przez PCA do 100. |
| **Gabor** | Gabor Filters | Filtry Gabora | Bank 32 filtrów (8 orientacji × 4 częstotliwości) wzorowanych na receptorach kory wzrokowej. Każdy filtr to iloczyn sinusoidy i gaussianu — wykrywa tekstury w określonym kierunku i skali. Z każdego filtra pobierane są: średnia i odchylenie standardowe odpowiedzi → 64 cechy. |
| **Minutiae** | Lip Ridge Minutiae | Minucje bruzd ust | Metoda zapożyczona z daktyloskopii. Po binaryzacji i szkieletyzacji obrazu obliczana jest liczba skrzyżowań linii (Crossing Number, CN). CN=1 → zakończenie, CN=3 → rozwidlenie, CN≥4 → skrzyżowanie. Łącznie 7 cech statystycznych. |

---

### Miary podobieństwa (Metoda 1)

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **SSIM** | Structural Similarity Index | Wskaźnik strukturalnego podobieństwa | Porównuje dwa obrazy (przeskalowane do 256×128 px, grayscale) pod kątem luminancji, kontrastu i struktury. Wynik ∈ [−1, 1]; wartość 1 oznacza identyczne obrazy. Lepszy niż MSE bo uwzględnia lokalne zależności przestrzenne. |
| **ORB** | Oriented FAST and Rotated BRIEF | Orientowane deskryptory punktów kluczowych | Wykrywa punkty kluczowe (narożniki, krawędzie) algorytmem FAST, a następnie opisuje je deskryptorami BRIEF. Dopasowuje pary punktów między obrazami (Brute-Force Matcher). Score = liczba dopasowań / łączna liczba punktów ∈ [0, 1]. |
| **Hist** | Histogram Correlation | Korelacja histogramów | Porównuje znormalizowane histogramy jasności obu obrazów (256 binów). Metoda korelacji Pearsona — wynik ∈ [−1, 1]. Szybka, ale wrażliwa na zmiany oświetlenia. |
| **Combined** | Combined Score (Weighted Combination) | Kombinacja ważona | Ważona średnia trzech powyższych miar: SSIM×0.5 + ORB×0.25 + Hist×0.25. Wartości SSIM i Hist normalizowane z [−1,1] → [0,1] przed sumowaniem. |

---

### Klasyfikatory

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **SVM** | Support Vector Machine | Maszyna wektorów nośnych | Klasyfikator wyznaczający hiperpłaszczyznę maksymalnego marginesu między klasami. Używa jądra RBF (Radial Basis Function) do mapowania cech do wymiaru, gdzie klasy są liniowo separowalne. Parametry: C=10 (kara za błędy), gamma='scale'. Probabilistyka przez skalowanie Platta (OvO — 231 binarnych klasyfikatorów dla 22 klas). |
| **RF** | Random Forest | Las losowy | Ensemble 200 niezależnych drzew decyzyjnych, każde trenowane na losowym podzbiorze danych (bagging) i losowym podzbiorze cech (sqrt(181) ≈ 13 cech/węzeł). Wynik to głosowanie większościowe. Odporny na przeuczenie, daje naturalny ranking ważności cech. Parametr `class_weight='balanced'` koryguje nierównoliczne klasy. |
| **k-NN** | k-Nearest Neighbors | k Najbliższych sąsiadów | Klasyfikuje obraz przez znalezienie k=15 najbliższych obrazów treningowych w przestrzeni 181-wymiarowej (miara Euklidesowa). Prawdopodobieństwo klasy obliczane własnym algorytmem: softmax z minimalnych odległości do każdej klasy (funkcja `_knn_softmax_proba`), nie standardowe głosowanie. |

---

### Techniki ML i oceny jakości

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **PCA** | Principal Component Analysis | Analiza głównych składowych | Metoda redukcji wymiarowości: znajduje kierunki (główne składowe) o największej wariancji danych. HOG daje 16 740 cech — PCA redukuje je do 100 najważniejszych składowych. Dopasowywany wyłącznie na zbiorze treningowym (bez danych testowych). |
| **CV** | Cross-Validation | Walidacja krzyżowa | Technika oceny modelu: dane dzielone są na K fragmentów (foldów). Model trenowany K razy, za każdym razem jeden fold jest zbiorem testowym, pozostałe K−1 treningowym. Daje rzetelną ocenę bez przeuczenia. |
| **5-fold CV** | 5-fold Stratified Cross-Validation | 5-foldowa stratyfikowana walidacja krzyżowa | Walidacja krzyżowa z K=5. Słowo „stratified" oznacza, że każdy fold zachowuje proporcje klas z całego datasetu. Seed=42 zapewnia powtarzalność podziałów. |
| **F1** | F1-score (macro-averaged) | Miara F1 (makro uśredniona) | Harmoniczna średnia precyzji i czułości: F1 = 2·(P·R)/(P+R). „Macro" oznacza, że F1 jest obliczane osobno dla każdej z 22 klas, a następnie uśredniane (każda klasa liczy się jednakowo, niezależnie od liczności). |
| **Accuracy** | Top-1 Accuracy | Dokładność (top-1) | Procent obrazów, dla których klasyfikator wskazał poprawną osobę na pierwszym miejscu (najwyższe prawdopodobieństwo). Baseline losowy dla 22 klas: 1/22 ≈ 4.5%. |
| **Top-3** | Top-3 Accuracy | Dokładność top-3 | Procent obrazów, dla których poprawna osoba znalazła się wśród 3 kandydatów z najwyższym prawdopodobieństwem. Ważna miara dla zastosowań asystujących (np. przeszukiwanie bazy). |
| **Top-5** | Top-5 Accuracy | Dokładność top-5 | Jak Top-3, ale wśród 5 kandydatów. Wyższe wartości Top-5 przy niskim Top-1 sugerują, że model „wie" o kim mowa, ale nie jest wystarczająco pewny. |
| **Confusion Matrix** | Confusion Matrix | Macierz pomyłek | Macierz 22×22, gdzie wiersz to prawdziwa klasa, kolumna to przewidywana. Przekątna (zielona) to poprawne identyfikacje. Elementy poza przekątną (czerwone) to błędy — można z nich odczytać które osoby są ze sobą mylone. |

---

### Preprocessing i przetwarzanie obrazów

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization | Adaptacyjne wyrównanie histogramu z ograniczeniem kontrastu | Wzmacnia lokalny kontrast obrazu dzieląc go na kafle 8×8 px i wyrównując histogram każdego kafla osobno. Parametr `clipLimit=2.0` zapobiega nadmiernemu wzmocnieniu szumu. Stosowany po konwersji do grayscale. |
| **Bilateral Filter** | Bilateral Filter | Filtr bilateralny | Wygładza obraz zachowując krawędzie (w przeciwieństwie do rozmycia Gaussowskiego). Uwzględnia zarówno podobieństwo przestrzenne jak i wartości pikseli. Parametry: d=9 (rozmiar okna), σ=75 (obie składowe). Usuwa szum aparatu fotograficznego. |
| **Grayscale** | Grayscale conversion | Konwersja do skali szarości | Przekształcenie obrazu RGB (3 kanały) do jednokanałowego obrazu jasności. Wzór: Y = 0.299·R + 0.587·G + 0.114·B (standard BT.601). Zmniejsza wymiarowość i uniezależnia od koloru. |
| **CN** | Crossing Number | Liczba skrzyżowań | Miara topologiczna ze szkieletyzacji odcisków: dla każdego piksela szkieletu zlicza ile razy zmienia się wartość (0/1) wśród 8 sąsiadów. CN=1 → zakończenie linii, CN=2 → zwykły piksel, CN=3 → rozwidlenie, CN≥4 → skrzyżowanie. |
| **1-NN** | 1-Nearest Neighbor | 1 Najbliższy sąsiad | Uproszczony wariant k-NN z k=1. W Metodzie 1: dla obrazu testowego wybierana jest osoba, której zdjęcie treningowe osiągnęło najwyższy score podobieństwa (SSIM / ORB / Hist / Combined). |

---

### Inne pojęcia

| Skrót | Pełna nazwa angielska | Polskie tłumaczenie | Wyjaśnienie |
|---|---|---|---|
| **GUI** | Graphical User Interface | Graficzny interfejs użytkownika | Aplikacja z interfejsem okienkowym/przeglądarkowym — w tym projekcie zbudowana w Streamlit (uruchamiana jako `streamlit run apps/app.py`). |
| **SHA-256** | Secure Hash Algorithm 256-bit | 256-bitowy bezpieczny skrót | Funkcja skrótu kryptograficznego — tutaj używana do obliczenia „odcisku" całego datasetu. Zmiana dowolnego zdjęcia zmienia hash → automatyczny retrain modeli. |
| **joblib** | joblib (Python library) | Biblioteka joblib | Biblioteka Python do serializacji dużych obiektów numpy i równoległego wykonywania. Używana do zapisu/wczytywania wytrenowanych modeli (`.joblib`). |
| **Platt scaling** | Platt scaling / Platt calibration | Kalibracja Platta | Metoda konwersji wyników SVM (odległości od hiperpłaszczyzny) na prawdopodobieństwa klas przez dopasowanie funkcji logistycznej. Włączana parametrem `probability=True` w sklearn. |
| **OvO** | One-vs-One | Jeden-przeciw-jednemu | Strategia wieloklasowa dla SVM: trenuje osobny klasyfikator binarny dla każdej pary klas. Dla 22 klas: C(22,2) = 231 klasyfikatorów. Finalna klasa wyłaniana przez głosowanie. |
| **bagging** | Bootstrap Aggregating | Agregacja bootstrapowa | Technika ensemble: każde drzewo w Random Forest trenowane na losowej próbce ze zwracaniem (bootstrap). Różne drzewa „widzą" różne dane → mniejsza korelacja błędów → lepsza generalizacja. |
| **M1** | Method 1 | Metoda 1 | Skrót używany w kodzie i wynikach: tradycyjne porównanie obrazów (SSIM / ORB / Hist / Combined + 1-NN). |
| **M2** | Method 2 | Metoda 2 | Skrót używany w kodzie i wynikach: klasyczne uczenie maszynowe na wektorze 181 cech (LBP + HOG→PCA + Gabor + Minutiae → SVM / RF / k-NN). |
