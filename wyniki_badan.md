# Wyniki wstępnych badań — Identyfikacja biometryczna na podstawie fotografii ust

**Data eksperymentu:** 2026-05-11 / 2026-05-12  
**Pliki źródłowe:**
- `results/experiment_20260512_000916.xlsx` — Metoda 2 (ML)
- `results/method1_cv.xlsx` — Metoda 1 (tradycyjna)

**Metoda ewaluacji:** 5-fold Stratified Cross-Validation (seed=42)

---

## 1. Parametry eksperymentu

| Parametr | Wartość |
|---|---|
| Liczba osób (klas) | 22 |
| Łączna liczba zdjęć | 338 |
| Średnio zdjęć na osobę | ~15–16 |
| Podział CV | 5-fold stratified (~270 train / ~68 test per fold) |
| Cechy wejściowe (M2) | LBP(10) + HOG→PCA(100) + Gabor(64) + Minutiae(7) = **181 cech** |
| HOG PCA komponenty | 100 |
| Klasyfikatory M2 | SVM (RBF, C=10), Random Forest (200 drzew), k-NN (k=15, softmax) |
| Miary M1 | SSIM, ORB, Histogram, Combined (wagi 0.5/0.25/0.25) |
| Baseline losowy | 1/22 = **4.55%** |

---

## 2. Wyniki zbiorcze — obie metody

### Pełne zestawienie (posortowane wg accuracy)

| # | Metoda | Klasyfikator/Miara | Accuracy (mean ± std) | Top-3 | Top-5 | F1 macro |
|---|---|---|---|---|---|---|
| 1 | **M1** | Combined | **90.24% ± 1.48%** | 94.97% | 95.56% | — |
| 2 | **M2** | SVM | **84.33% ± 1.69%** | 92.62% | 95.27% | 83.89% |
| 3 | **M2** | Random Forest | **84.33% ± 2.86%** | 92.01% | 93.78% | 83.23% |
| 4 | **M1** | ORB | 83.73% ± 5.89% | 89.95% | 91.44% | — |
| 5 | **M2** | k-NN | 77.81% ± 3.06% | 88.17% | 91.43% | 76.39% |
| 6 | **M1** | SSIM | 76.62% ± 2.38% | 88.16% | 90.82% | — |
| 7 | **M1** | Histogram | 30.50% ± 6.49% | 42.93% | 50.34% | — |
| — | *Baseline losowy* | — | *4.55%* | *13.64%* | *22.73%* | — |

---

## 3. Metoda 2 — wyniki per fold

### SVM (Support Vector Machine, jądro RBF, C=10)

| Fold | Accuracy | Top-3 | Top-5 | Precision | Recall | F1 | Czas [s] |
|---|---|---|---|---|---|---|---|
| 1 | 82.35% | 86.76% | 91.18% | 89.34% | 83.71% | 83.63% | 1.0 |
| 2 | 82.35% | 92.65% | 94.12% | 80.61% | 82.58% | 80.04% | 0.8 |
| 3 | 85.29% | 94.12% | 98.53% | 90.00% | 85.23% | 85.22% | 0.9 |
| 4 | **86.57%** | 94.03% | 94.03% | 88.64% | 86.74% | 86.31% | 0.7 |
| 5 | 85.07% | **95.52%** | **98.51%** | 87.12% | 85.61% | 84.27% | 0.8 |
| **MEAN ± STD** | **84.33% ± 1.69%** | **92.62%** | **95.27%** | — | — | **83.89% ± 2.13%** | — |

### Random Forest (200 drzew, max_features='sqrt', class_weight='balanced')

| Fold | Accuracy | Top-3 | Top-5 | Precision | Recall | F1 | Czas [s] |
|---|---|---|---|---|---|---|---|
| 1 | 82.35% | 89.71% | 92.65% | 88.96% | 83.71% | 84.32% | 1.1 |
| 2 | 85.29% | 92.65% | 92.65% | 80.08% | 81.82% | 78.87% | 1.2 |
| 3 | 82.35% | 94.12% | 97.06% | 86.44% | 83.33% | 83.06% | 1.1 |
| 4 | 82.09% | 91.04% | 92.54% | 84.85% | 80.68% | 81.07% | 1.3 |
| 5 | **89.55%** | 92.54% | 94.03% | **91.82%** | **89.39%** | **88.81%** | 1.1 |
| **MEAN ± STD** | **84.33% ± 2.86%** | **92.01%** | **93.78%** | — | — | **83.23% ± 3.35%** | — |

### k-NN (k=15, odległość euklidesowa, softmax z minimalnych odległości)

| Fold | Accuracy | Top-3 | Top-5 | Precision | Recall | F1 | Czas [s] |
|---|---|---|---|---|---|---|---|
| 1 | 79.41% | 85.29% | 88.24% | 84.79% | 79.55% | 78.43% | 0.7 |
| 2 | **72.06%** | 85.29% | 91.18% | 74.32% | 73.11% | 69.63% | 0.8 |
| 3 | 80.88% | 92.65% | 92.65% | 87.65% | 82.20% | 80.04% | 0.9 |
| 4 | 77.61% | 85.07% | 89.55% | 84.32% | 76.14% | 77.04% | 2.1 |
| 5 | 79.10% | 92.54% | 95.52% | 78.11% | 77.65% | 76.81% | 0.7 |
| **MEAN ± STD** | **77.81% ± 3.06%** | **88.17%** | **91.43%** | — | — | **76.39% ± 3.57%** | — |

---

## 4. Metoda 1 — wyniki per fold

### SSIM (Structural Similarity Index, 1-NN)

| Fold | Accuracy | Top-3 | Top-5 | Czas [s] |
|---|---|---|---|---|
| 1 | **80.88%** | 89.71% | 92.65% | 66.3 |
| 2 | 75.00% | 86.76% | 91.18% | 77.1 |
| 3 | 75.00% | 88.24% | 91.18% | 71.5 |
| 4 | 74.63% | 85.07% | 86.57% | 70.2 |
| 5 | 77.61% | 91.04% | 92.54% | 70.4 |
| **MEAN ± STD** | **76.62% ± 2.38%** | **88.16%** | **90.82%** | ~71 s |

### ORB (Oriented FAST and Rotated BRIEF, 1-NN)

| Fold | Accuracy | Top-3 | Top-5 | Czas [s] |
|---|---|---|---|---|
| 1 | 85.29% | 86.76% | 88.24% | 242.5 |
| 2 | 80.88% | 89.71% | 89.71% | 219.0 |
| 3 | 85.29% | 89.71% | 91.18% | 247.7 |
| 4 | 74.63% | 88.06% | 91.04% | 250.8 |
| 5 | **92.54%** | **95.52%** | **97.01%** | 389.8 |
| **MEAN ± STD** | **83.73% ± 5.89%** | **89.95%** | **91.44%** | ~270 s |

ORB jest **najbardziej niestabilny** (std=5.89%) — Fold 5 dał 92.54%, ale Fold 4 tylko 74.63%. Liczba wykrytych punktów kluczowych silnie zależy od oświetlenia zdjęcia.

### Histogram Correlation (korelacja histogramów jasności, 1-NN)

| Fold | Accuracy | Top-3 | Top-5 | Czas [s] |
|---|---|---|---|---|
| 1 | 25.00% | 39.71% | 45.59% | 87.7 |
| 2 | 20.59% | 30.88% | 35.29% | 73.8 |
| 3 | 36.76% | 47.06% | 55.88% | 74.7 |
| 4 | 34.33% | 49.25% | 53.73% | 65.9 |
| 5 | 35.82% | 47.76% | 61.19% | 70.6 |
| **MEAN ± STD** | **30.50% ± 6.49%** | **42.93%** | **50.34%** | ~74 s |

Histogram jest **najsłabszą metodą** — 30.50% to zaledwie 6.7× ponad baseline. Rozkład jasności pikseli nie opisuje struktury bruzd; jest wrażliwy na zmiany ekspozycji i oświetlenia.

### Combined Score (SSIM×0.5 + ORB×0.25 + Hist×0.25, 1-NN)

| Fold | Accuracy | Top-3 | Top-5 | Czas [s] |
|---|---|---|---|---|
| 1 | 89.71% | 95.59% | 97.06% | 394.2 |
| 2 | 88.24% | 91.18% | 92.65% | 423.0 |
| 3 | **91.18%** | **98.53%** | **98.53%** | 399.3 |
| 4 | 89.55% | 95.52% | 95.52% | 428.4 |
| 5 | **92.54%** | 94.03% | 94.03% | 412.8 |
| **MEAN ± STD** | **90.24% ± 1.48%** | **94.97%** | **95.56%** | ~412 s |

Combined jest **najdokładniejszy i najbardziej stabilny** (std=1.48%) spośród wszystkich metod. Łączenie trzech niezależnych sygnałów redukuje błędy przypadkowe każdej z miar.

---

## 5. Analiza macierzy pomyłek — SVM (suma 5 foldów, 338 obrazów)

### Osoby z najwyższą skutecznością

| Osoba | Poprawne | Łącznie | Skuteczność | Uwagi |
|---|---|---|---|---|
| **Nr_02** | 14 | 14 | **100.0%** | Zero błędów |
| **Nr_08** | 15 | 15 | **100.0%** | Zero błędów |
| **Nr_07** | 17 | 18 | 94.4% | 1 błąd → Nr_21 |
| **Nr_11** | 17 | 18 | 94.4% | 1 błąd → Nr_07 |
| **Nr_12** | 17 | 18 | 94.4% | 1 błąd → Nr_02 |
| **Nr_21** | 17 | 18 | 94.4% | 1 błąd → Nr_18 |
| **Nr_14** | 16 | 17 | 94.1% | 1 błąd → Nr_22 |
| **Nr_22** | 16 | 17 | 94.1% | 1 błąd → Nr_21 |

### Osoby z najniższą skutecznością

| Osoba | Poprawne | Łącznie | Skuteczność | Z kim mylona |
|---|---|---|---|---|
| **Nr_19** | 9 | 17 | **52.9%** | 2→Nr_07, 2→Nr_12, 2→Nr_16, 1→Nr_20, 1→Nr_21 |
| **Nr_13** | 5 | 8 | **62.5%** | 1→Nr_06, 1→Nr_15, 1→Nr_20 |
| **Nr_03** | 8 | 13 | **61.5%** | 3→Nr_12, 1→Nr_16, 1→Nr_19 |
| **Nr_16** | 10 | 16 | **62.5%** | 1→Nr_07, 1→Nr_08, 1→Nr_09, 1→Nr_12, 1→Nr_15, 1→Nr_19 |
| **Nr_17** | 9 | 14 | **64.3%** | 2→Nr_15, 1→Nr_04, 1→Nr_06, 1→Nr_20 |

### Najczęstsze pary pomyłek

| Para | Błędów | Uwagi |
|---|---|---|
| **Nr_03 → Nr_12** | 3 | Największa jednostronna pomyłka — podobny wzorzec bruzd |
| **Nr_19 → Nr_07/12/16** | po 2 | Osoba 19 ma rozproszony wzorzec — mylona z wieloma klasami |
| **Nr_17 → Nr_15** | 2 | Zbliżone wzorce bruzd |

---

## 6. Porównanie czasów działania

| Metoda | Czas treningu per fold | Czas testu (68 obrazów) | Czas pełnego CV (5 foldów) |
|---|---|---|---|
| M2 SVM | ~0.8 s | ~0.8 s | **~8 s** |
| M2 Random Forest | ~1.2 s | ~1.2 s | **~12 s** |
| M2 k-NN | ~0.8 s | ~0.8 s | **~8 s** |
| M1 SSIM | brak treningu | ~71 s | **~6 min** |
| M1 ORB | brak treningu | ~270 s | **~22 min** |
| M1 Histogram | brak treningu | ~74 s | **~6 min** |
| M1 Combined | brak treningu | ~412 s | **~34 min** |

SVM jest **~250× szybszy** od Combined przy porównywalnym lub tylko nieznacznie niższym wyniku.

---

## 7. Pliki wyników

| Plik | Arkusze | Zawartość |
|---|---|---|
| `results/experiment_20260512_000916.xlsx` | Run_info, Predictions, Summary_svm/rf/knn, CM_svm/rf/knn | Pełna ewaluacja ML (1014 predykcji) |
| `results/method1_cv.xlsx` | Run_info, Predictions, Summary_ssim/orb/hist/combined | Pełna ewaluacja tradycyjna (1352 predykcji) |
| `results/plots/confusion_matrix_svm.png` | — | Heatmapa macierzy pomyłek SVM |
| `results/plots/confusion_matrix_rf.png` | — | Heatmapa macierzy pomyłek RF |
| `results/plots/confusion_matrix_knn.png` | — | Heatmapa macierzy pomyłek k-NN |

---

---

# Podsumowanie wyników — dla promotora

## Co było celem badania?

Celem było sprawdzenie, czy można automatycznie rozpoznać osobę na podstawie fotografii ust (cheiloskopia), oraz porównanie dwóch różnych podejść do tego problemu. Zbiór danych to 338 zdjęć 22 różnych osób. Gdybyśmy zgadywali losowo, trafilibyśmy poprawnie w około 4,5% przypadków (1 z 22 klas).

---

## Co zrobiono?

Zaimplementowano i przetestowano dwa niezależne podejścia:

**Metoda 1 — bezpośrednie porównanie zdjęć:**
Każde zdjęcie testowe porównywane jest ze wszystkimi zdjęciami treningowymi za pomocą miar podobieństwa obrazów. Wygrywa osoba, której zdjęcie jest „najbardziej podobne". Przetestowano 4 miary: SSIM (strukturalne podobieństwo), ORB (punkty kluczowe), Histogram jasności i ich kombinację ważoną.

**Metoda 2 — uczenie maszynowe na cechach biometrycznych:**
Z każdego zdjęcia wyodrębniane są 181 liczb opisujących bruzdy ust (tekstura LBP, gradienty HOG, filtry Gabora, minucje bruzd). Na tych liczbach trenowane są trzy klasyfikatory: SVM, Random Forest i k-NN.

Każda metoda oceniana była metodą 5-krotnej walidacji krzyżowej — dane dzielone na 5 części, każda raz służy jako zbiór testowy.

---

## Co wyszło?

### Wyniki jednym zdaniem na metodę:

| Metoda | Accuracy | Interpretacja |
|---|---|---|
| Combined (M1) | **90,2%** | Najlepsza dokładność — ale bardzo wolna |
| SVM (M2) | **84,3%** | Prawie tak samo dobra, 250× szybsza |
| Random Forest (M2) | **84,3%** | Identyczny wynik jak SVM |
| ORB (M1) | 83,7% | Dobra, ale niestabilna (±5,9%) |
| k-NN (M2) | 77,8% | Wyraźnie słabsza od SVM/RF |
| SSIM (M1) | 76,6% | Umiarkowana skuteczność |
| Histogram (M1) | **30,5%** | Praktycznie bezużyteczna |
| *Losowe zgadywanie* | *4,5%* | — |

### Najważniejsze odkrycie

**Metoda bezpośredniego porównania zdjęć (Combined, 90,2%) okazała się dokładniejsza niż uczenie maszynowe (SVM/RF, 84,3%).** Jest to wynik nieoczekiwany — zazwyczaj uczenie maszynowe przewyższa proste metody porównawcze. Wyjaśnienie jest następujące: bezpośrednie porównanie obrazów zachowuje pełną przestrzenną strukturę wzorców bruzd, podczas gdy ML widzi jedynie 181 zagregowanych liczb, tracąc część informacji o rozmieszczeniu przestrzennym.

**Jednak** Combined potrzebuje ~34 minuty na pełny test, a SVM — zaledwie ~8 sekund. Różnica: 250-krotna. Dla systemu działającego w czasie rzeczywistym jedyną praktyczną opcją jest ML.

---

## Co oznaczają poszczególne metryki?

**Accuracy (dokładność)** — procent osób wskazanych poprawnie na pierwszym miejscu. SVM: 84,3% oznacza, że na 68 zdjęciach testowych system poprawnie wskazał osobę w ~57 przypadkach.

**Top-5 Accuracy** — procent przypadków, gdzie prawidłowa osoba znalazła się wśród 5 wskazanych kandydatów. SVM Top-5 = 95,3% oznacza, że w 95 na 100 zdjęć osoba jest w pierwszej piątce. To bardzo ważna metryka dla systemu wspomagającego eksperta (kryminalistyka) — ekspert przegląda 5 kandydatów i wybiera właściwego.

**F1 macro** — uśredniona miara łącząca precyzję i czułość dla każdej z 22 klas osobno. SVM F1=83,9% oznacza, że system działa równomiernie dobrze dla wszystkich klas — żadna klasa nie jest kompletnie ignorowana.

**Std (odchylenie standardowe)** — miara stabilności. SVM std=1,69% oznacza, że wyniki w 5 testach różniły się co najwyżej o ~2 punkty procentowe — system jest przewidywalny. ORB std=5,89% oznacza, że wyniki skaczą mocno między testami — system jest niestabilny.

---

## Co sprawdziło się, co nie?

✅ **Działało dobrze:**
- Wszystkie metody wielokrotnie biją losowe zgadywanie (4,5%) — bruzdy ust rzeczywiście zawierają informację biometryczną
- SVM i RF są stabilne i szybkie — nadają się do aplikacji
- Top-5 powyżej 91% dla wszystkich sensownych metod — system jest użyteczny jako narzędzie wspomagające eksperta
- 2 osoby (Nr_02, Nr_08) identyfikowane ze 100% skutecznością przez SVM

❌ **Nie zadziałało:**
- Histogram jasności (30,5%) — za prosta miara, nieodporna na zmiany oświetlenia
- ORB jest niestabilny (std=5,89%) — liczba wykrytych punktów kluczowych zależy od warunków zdjęcia
- 2 osoby (Nr_19: 52,9%, Nr_13: 62,5%) trudne do identyfikacji — możliwe podobieństwo biologiczne wzorców lub duża zmienność zdjęć tej samej osoby

---

## Kto jest trudny do rozpoznania i dlaczego?

**Nr_19** (52,9% SVM) — mylona aż z 5 różnymi osobami, rozsiana po całej macierzy. Prawdopodobna przyczyna: duża zmienność wyrazu ust między sesjami zdjęciowymi lub bruzdy niewyróżniające się spośród 22 klas.

**Nr_13** (62,5% SVM) — mała liczba zdjęć testowych (8 sztuk), 3 błędy rozłożone na różne klasy. Mały rozmiar klasy utrudnia wnioskowanie statystyczne.

**Nr_03 mylona z Nr_12** — 3 razy SVM klasyfikuje Nr_03 jako Nr_12. To sugeruje rzeczywiste biologiczne podobieństwo wzorców bruzd tych dwóch osób (ewentualnie podobny wiek lub cechy twarzy). Warto sprawdzić dane demograficzne.

---

## Wnioski końcowe

1. **Cheiloskopia jest wykonalna biometrycznie** — system bez żadnej wiedzy wstępnej, ucząc się wyłącznie z 270 zdjęć, potrafi rozpoznać osobę spośród 22 z 84–90% skutecznością.

2. **Metoda 1 (Combined) jest najdokładniejsza, ale niepraktyczna** — 90,2% przy czasie 34 minut na CV jest do przyjęcia w badaniach laboratoryjnych, ale wyklucza zastosowania czasu rzeczywistego.

3. **Metoda 2 (SVM/RF) to kompromis dokładność–szybkość** — 84,3% przy czasie 8 sekund to wynik praktycznie użyteczny w systemie biometrycznym.

4. **Top-5 powyżej 95% dla najlepszych metod** — system wchodzi w zastosowania wspomagające eksperta kryminalistycznego: zamiast przeszukiwać bazę ręcznie, ekspert dostaje 5 kandydatów i wybiera właściwego.

5. **Histogram jasności nie nadaje się do identyfikacji biometrycznej bruzd ust** — 30,5% to wynik nieakceptowalny, potwierdzający że sama jasność pikseli jest zbyt prymitywną cechą.

6. **Dalsze kierunki:** zbadanie osób trudnych (Nr_19, Nr_03), analiza wpływu LampaON/LampaOFF na jakość identyfikacji, ewentualne porównanie z sieciami neuronowymi jako Metodą 3.
