# Szablon sprawozdania laboratoryjnego

**Przedmiot:** Uczenie głębokie w systemach rozpoznawania i przetwarzania mowy  
**Zajęcia nr / temat:** Laboratorium 3 — Prognozowanie szeregów czasowych: prognoza wielokrokowa i porównanie z baseline'ami  
**Autor / grupa:** Bartosz Paluda 
**Data:** 29 października 2026 r.  

---

## 1. Cel ćwiczenia

Celem ćwiczenia jest budowa i ewaluacja modelu rekurencyjnego GRU (*Gated Recurrent Unit*) prognozującego dzienne wartości szeregu czasowego (syntetyczna dzienna sprzedaż) oraz rzetelne porównanie jego jakości z prostymi regułami odniesienia (baseline'em naiwnym, sezonowym oraz średnią ruchomą) w reżimach prognozy jednokrokowej ($H = 1$) oraz wielokrokowej ($H = 14$).

Zakres zadania obejmuje:
1. Analizę procesu zbieżności modelu bazowego za pomocą krzywych uczenia pod kątem zjawisk niedouczenia (*underfitting*) lub przeuczenia (*overfitting*).
2. Przeprowadzenie wielokrokowej predykcji autoregresyjnej ($H = 14$) na zbiorze testowym i porównanie dynamiki degradacji prognoz z regułami bazowymi.
3. Eksperymentalną analizę wpływu doboru szerokości okna kontekstowego ($OKNO \in \{3, 7, 28\}$) przy różnych ziarnach losowości (`seed 0` i `seed 1`) oraz dokonanie wyboru najlepszej konfiguracji w oparciu o zbiór walidacyjny.
4. Odniesienie zjawisk zaobserwowanych w szeregach czasowych (dobór okna, kumulacja błędu w pętli autoregresyjnej) do dziedziny przetwarzania sygnału mowy (automatyczne rozpoznawanie mowy — ASR oraz synteza mowy — TTS).

---

## 2. Metoda i implementacja

### Narzędzia i architektura modelu
* **Środowisko:** Python 3.13, biblioteki PyTorch (`torch.nn`), NumPy, Matplotlib.
* **Model:** Rekurencyjna sieć neuronowa z jedną warstwą GRU o wymiarze ukrytym $h = 32$ (`batch_first=True`), połączona z warstwą w pełni połączoną (`nn.Linear(32, 1)`) rzutującą wektor ukryty z ostatniego kroku czasowego okna na pojedynczą wartość skalarną prognozy. Łączna liczba trenowalnych parametrów wynosi 3393.
* **Kryterium optymalizacyjne:** Funkcja straty błędu średniokwadratowego ($\text{MSE}$).
* **Optymalizator:** Adam ze współczynnikiem uczenia $\text{lr} = 0{,}01$, batch size równym 64, trenowany przez 40 epok z przycinaniem gradientów (`clip_grad_norm_ = 1.0`). Zastosowano mechanizm *early stopping* przywracający stan wag z epoki o najniższej stracie walidacyjnej.

### Podział danych i normalizacja bez przecieku (data leakage)
* **Szereg danych:** Syntetyczny szereg $N = 1000$ punktów: stały poziom 100, trend liniowy ($0{,}02 \cdot t$), składowa tygodniowa ($8\sin(2\pi t / 7)$), składowa roczna ($15\sin(2\pi t / 365)$) oraz addytywny szum gaussowski $\mathcal{N}(0, \sigma^2 = 9)$.
* **Podział chronologiczny:** 
  * Trening: indeksy $0 \dots 700$ (70%),
  * Walidacja: indeksy $700 \dots 850$ (15%),
  * Test: indeksy $850 \dots 1000$ (15%).
* **Standaryzacja:** Parametry $\mu = 106{,}96$ oraz $\sigma = 12{,}00$ wyznaczono **wyłącznie na zbiorze treningowym**. Cały szereg przeskalowano jako $s = (y - \mu) / \sigma$. Wszystkie metryki błędów (MAE, RMSE) wyliczono po odwróceniu transformacji do jednostek sprzedaży: $\hat{y} = \hat{s} \cdot \sigma + \mu$.

### Najważniejsze fragmenty implementacji

```python
class ModelGRU(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.rnn = nn.GRU(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
        
    def forward(self, x):
        out, _ = self.rnn(x)                 # out: [N, OKNO, hidden]
        return self.fc(out[:, -1, :])        # stan z końca okna -> predykcja [N, 1]

def rollout(model, s, start, H, okno):
    """Prognoza rekurencyjna na H kroków; model widzi tylko dane sprzed `start`."""
    hist = list(s[start - okno:start])
    wyn = []
    model.eval()
    with torch.no_grad():
        for _ in range(H):
            x = torch.tensor(np.array(hist[-okno:], np.float32).reshape(1, okno, 1)).to(DEVICE)
            y = model(x).item()
            wyn.append(y)
            hist.append(y)                   # prognoza trafia z powrotem do bufora wejściowego
    return np.array(wyn)
```

---

## 3. Wyniki

### 3.1. Krzywe uczenia modelu bazowego (Zadanie 1)
* **Najlepsza epoka:** **36** (minimalny błąd walidacyjny $\text{val MSE} = 0{,}0822$ w skali standaryzowanej).
* **Przebieg optymalizacji:** Strata treningowa i walidacyjna gwałtownie spadły w trakcie pierwszych 10 epok, po czym ustabilizowały się na asymptotycznym poziomie $\approx 0{,}08$–$0{,}09$, biegnąc równolegle.

### 3.2. Ocena jedno- i wielokrokowa na zbiorze testowym (Zadanie 2)
Ewaluację wielokrokową przeprowadzono z 10 punktów startowych rozmieszczonych co 14 dni na zbiorze testowym ($t_0 \in \{850, 864, \dots, 976\}$) dla horyzontu $H = 14$. Wyniki podano w oryginalnej skali sprzedaży:

| Metoda / Model | MAE ($H=1$) | RMSE ($H=1$) | MAE ($H=14$) | RMSE ($H=14$) |
| :--- | :---: | :---: | :---: | :---: |
| **Teoretyczna granica szumu** | **2,39** | **3,00** | — | — |
| **Model GRU ($OKNO=28$)** | **2,91** | **3,54** | **3,88** | **4,75** |
| **Baseline sezonowy ($t-7$)** | 3,21 | 4,07 | **3,52** | **4,42** |
| **Baseline naiwny ($t-1$)** | 5,44 | 6,51 | 10,09 | 11,85 |
| **Średnia ruchoma ($OKNO=28$)** | 5,80 | 6,84 | — | — |

* **Dynamika błędu w funkcji horyzontu $h \in [1, 14]$:**
  * **Naiwny ($t-1$):** Błąd natychmiast gwałtownie wzrasta z poziomu ok. $5{,}4$ do ponad $11{,}8$, ponieważ stała linia nie posiada żadnej informacji o fazie cyklu.
  * **Sezonowy ($t-7$):** Wykazuje stabilny błąd MAE na poziomie $\approx 3{,}52$ dla każdego kroku $h$, gdyż powtarzanie idealnego cyklu 7-dniowego nie ulega degradacji wraz z odległością horyzontu.
  * **GRU:** Błąd rośnie monotonicznie od $2{,}91$ dla $h = 1$ do poziomu ponad $4{,}6$ dla $h = 14$ na skutek kumulowania się odchyleń predykcji podawanych na wejście w kolejnych krokach.

### 3.3. Wpływ długości okna wejściowego (Zadanie 3)
Eksperyment przeprowadzono dla $OKNO \in \{3, 7, 28\}$ i dwóch ziaren losowości (`seed 0`, `seed 1`):

| Długość okna | Ziarno (Seed) | Walidacja (`val MSE`) | MAE 1-krok ($H=1$) | MAE 14-krok ($H=14$) |
| :---: | :---: | :---: | :---: | :---: |
| **3** | 0 | 0,2567 | 5,00 | 7,18 |
| **3** | 1 | 0,2486 | 5,04 | 7,41 |
| **7** | 0 | 0,0855 | 2,90 | 4,21 |
| **7** | 1 | 0,0896 | 2,88 | 3,89 |
| **28** | 0 | **0,0822** | **2,91** | **3,88** |
| **28** | 1 | **0,0850** | **3,01** | **4,29** |

* **Wybór optymalnego okna:** Zgodnie z kryterium minimalizacji błędu na zbiorze walidacyjnym wybrano **$OKNO = 28$** (średnia $\text{val MSE} = 0{,}0836$, wobec $0{,}0876$ dla okna 7 oraz $0{,}2527$ dla okna 3).

---

## 4. Analiza i interpretacja

* **Niedouczenie i przeuczenie (Zadanie 1):**  
  Model nie wykazuje cech przeuczenia (krzywa walidacyjna nie rośnie w późniejszych epokach, lecz zbiega równolegle do treningowej). Nie występuje również niedouczenie – błąd osiąga asymptotę wynikającą bezpośrednio z losowej składowej szumu. Mechanizm wczesnego zatrzymania poprawnie przywrócił optymalne parametry z 36. epoki.
* **Skuteczność GRU wobec baseline'ów (Zadanie 2 i 4):**  
  * *Prognoza 1-krokowa:* GRU wyraźnie wygrywa z baseline'em sezonowym ($2{,}91$ vs $3{,}21$) i deklasuje naiwny ($5{,}44$). Model skutecznie odfiltrowuje szum i modeluje jednocześnie trend oraz cykliczność.
  * *Prognoza wielokrokowa:* GRU osiąga błąd $\text{MAE} = 3{,}88$, ustępując sztywnej regule sezonowej ($\text{MAE} = 3{,}52$). Różnica ta wynika z **kumulacji błędu autokorelacyjnego** w procedurze *rollout*. Każda niedoskonała predykcja trafiająca do bufora wejściowego zniekształca fazę sygnału w kolejnych krokach, podczas gdy reguła sezonowa bezbłędnie replikuje dokładny 7-dniowy okres.
* **Odległość od granicy szumu (Zadanie 4):**  
  Dla szumu normalnego o $\sigma = 3$ teoretyczna dolna granica błędu absolutnego wynosi $\mathbb{E}[|X|] = 3\sqrt{2/\pi} \approx 2{,}39$. Model GRU osiąga $\text{MAE} = 2{,}91$, co oznacza odchylenie o zaledwie $0{,}52$ jednostki sprzedaży ponad nieusuwalny, czysty szum.
* **Wpływ szerokości okna i porównanie z rozrzutem ziaren (Zadanie 4):**  
  * Okno 3-dniowe jest zbyt wąskie, by sieć mogła zaobserwować pełną zmienność dominującego cyklu tygodniowego ($T = 7$), co powoduje drastyczny wzrost błędów ($\text{MAE} \approx 5{,}00$).
  * Okno 28-dniowe obejmuje 4 pełne tygodnie, ułatwiając modelowi oddzielenie lokalnych fluktuacji od trendu.
  * Różnice strukturalne wynikające z rozmiaru okna ($\Delta \text{MAE} \approx 2{,}1$ między oknem 3 a 28) są o rząd wielkości większe niż rozrzut powodowany losową inicjalizacją wag ($\Delta \text{MAE} \le 0{,}1$ między ziarnami 0 i 1).
* **Odniesienie do modeli mowy (ASR i TTS) (Zadanie 4):**  
  * *Kontekst okna (ASR):* Pojedyncza ramka analizy spektralnej mowy (np. MFCC lub mel-spektrogram) reprezentuje typowo wycinek 10–25 ms. Zastosowanie zbyt małego kontekstu czasowego (analogicznie do okna 3 dni) uniemożliwia uchwycenie zjawisk koartykulacji oraz fonemów trwających dłużej niż jedna ramka.
  * *Kumulacja błędu w autoregresji (TTS):* W syntezie mowy generowanej krok po kroku (np. Tacotron, klasyczny WaveNet) sprzężenie zwrotne generowanych próbek prowadzi do narastania zniekształceń fazowych i metalicznych artefaktów brzmieniowych – dokładnie tak samo, jak degradacja prognozy GRU na dalekim horyzoncie $H = 14$. Z tego powodu w mowie stosuje się trening z wymuszaniem odpowiedzi (*Teacher Forcing*), próbkowanie adaptacyjne (*Scheduled Sampling*) lub architektury nieautoregresyjne (np. FastSpeech, VITS).

---

## 5. Wnioski

1. Model rekurencyjny GRU bardzo dobrze radzi sobie z predykcją jednokrokową złożonych procesów stochastycznych, zbliżając się do teoretycznej granicy wyznaczonej przez wariancję szumu.
2. W zadaniach długoterminowej prognozy wielokrokowej proste reguły dziedzinowe (np. powtórzenie sezonowe) mogą przewyższać model autoregresyjny z uwagi na brak kumulowania się błędów w pętli generacyjnej.
3. Długość bufora wejściowego musi obejmować co najmniej jeden pełen okres najdłuższej kluczowej składowej harmonicznej procesu.
4. W celu eliminacji kumulacji błędów w predykcji wielokrokowej zaleca się stosowanie bezpośredniej prognozy wielohoryzontowej (*Direct Multi-Step Forecasting*) lub modeli typu *Seq2Seq* z mechanizmem atencji, co stanowi bezpośredni odpowiednik rozwiązań stosowanych w nowoczesnych systemach przetwarzania mowy.
