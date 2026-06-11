# Zadanie 2 – Optymalizacja eliminacji Gaussa

## 1. Cel

Celem zadania była implementacja i optymalizacja algorytmu eliminacji Gaussa
(bez pivotingu) pod kątem architektury posiadanego procesora, weryfikacja
poprawności wyniku względem wersji referencyjnej oraz pomiar wydajności
w GFLOP/s.

## 2. Procesor

<img width="1034" height="291" alt="image" src="https://github.com/user-attachments/assets/65f81ff5-6f9e-42ca-ac67-cbf76671eab1" />


Pomiary wykonano na procesorze **Intel Core i5-8300H** (Coffee Lake):

| Parametr | Wartość |
|---|---|
| Rdzenie / wątki | 4 / 8 |
| Taktowanie | 2.3 GHz (turbo do 4.0 GHz) |
| Jednostki wektorowe | SSE, AVX, **AVX2**, **FMA** (rejestry 256-bit) |
| L1d | 32 KB na rdzeń |
| L2 | 256 KB na rdzeń |
| L3 | 8 MB wspólne |

Procesor obsługuje AVX2 z FMA, czyli na rejestrach 256-bitowych mieszczą się
4 liczby `double`, a instrukcja FMA wykonuje mnożenie i dodawanie w jednym
kroku. Teoretyczny szczyt jednego rdzenia to 2 jednostki FMA × 4 double ×
2 operacje = 16 FLOP/cykl, co przy 4.0 GHz daje ok. **64 GFLOP/s** (1 wątek,
double precision).

## 3. Implementacja

Kod znajduje się w pliku `ge.c` i zawiera dwie wersje algorytmu:

**Wersja referencyjna `ge_ref`** – klasyczna potrójna pętla:

```c
for (k = 0; k < SIZE; k++)
  for (i = k+1; i < SIZE; i++) {
    multipler = A[i][k] / A[k][k];
    for (j = k+1; j < SIZE; j++)
      A[i][j] = A[i][j] - A[k][j] * multipler;
  }
```

**Wersja zoptymalizowana `ge_opt`** – wzorowana na instruktażu
how-to-optimize-gemm oraz laboratoriach:

1. **Wektoryzacja AVX2 + FMA** – wewnętrzny kernel przetwarza po 8 elementów
   wiersza naraz (2 rejestry `__m256d`), aktualizacja `c = c - p*m` jest
   wykonywana jedną instrukcją `_mm256_fnmadd_pd` (FMA dostępne na i5-8300H).
2. **Aktualizacja blokowa rangi KC=4** – zamiast osobnej aktualizacji rangi 1
   dla każdego `k`, macierz resztkowa (trailing matrix) jest aktualizowana raz
   na KC kroków: `A22 = A22 - L21 * U12`. Dzięki temu każdy element `A22`
   jest ładowany z pamięci raz na 4 kroki zamiast za każdym razem (lepsza
   lokalność danych, więcej obliczeń na jeden dostęp do pamięci).
3. **Blokowanie pod cache** – aktualizacja jest dzielona na bloki
   MC×NC = 128×256. Blok pivot-wierszy (KC×NC = 8 KB) mieści się w L1,
   a blok roboczy (MC×NC = 256 KB doubli) w L2/L3.
4. **Packing** – wiersze U i mnożniki L są kopiowane do małych, ciągłych
   buforów, dzięki czemu kernel czyta dane sekwencyjnie.

Kompilacja: `gcc -O2 -mavx2 -mfma ge.c -o ge` (flagi dopasowane do
posiadanego procesora – AVX2 i FMA). Wszystkie pomiary jednowątkowe.

## 4. Weryfikacja poprawności

Program dla zadanego rozmiaru wypełnia dwie identyczne macierze losowymi
wartościami (`srand(1)`), wykonuje na jednej `ge_ref`, na drugiej `ge_opt`,
a następnie porównuje wszystkie elementy i wypisuje maksymalny błąd względny.
Wynik uznawany jest za poprawny, gdy błąd < 1e-6. Różnice rzędu 1e-11–1e-7
wynikają wyłącznie z innej kolejności operacji zmiennoprzecinkowych
(blokowanie + FMA), nie z błędu algorytmu. **Weryfikacja przeszła dla
wszystkich testowanych rozmiarów (200–2000).**

## 5. Szacowanie liczby operacji

Dla kroku `k` wykonuje się `(n-k-1)` dzieleń (mnożniki) oraz
`2*(n-k-1)^2` mnożeń i odejmowań. Po zsumowaniu po wszystkich `k`:

```
FLOP(n) = n(n-1)(2n-1)/3 + n(n-1)/2  ≈  (2/3) n^3
```

GFLOP/s liczone jest jako `FLOP(n) / czas / 1e9`.

## 6. Wyniki


<img width="1014" height="514" alt="image" src="https://github.com/user-attachments/assets/33d8a417-c8cc-4571-a65e-bce882f06cfe" />

## 7. Wnioski

- **Wersja zoptymalizowana jest 2–3 razy szybsza od referencyjnej**
  i osiąga do ~14 GFLOP/s, czyli ok. 22% teoretycznego szczytu rdzenia
  (64 GFLOP/s).
- **Eliminacja Gaussa ma niską intensywność arytmetyczną** – aktualizacja
  rangi 1 wykonuje tylko 2 FLOP na każdy załadowany i zapisany element,
  dlatego nie da się osiągnąć wysycenia FLOPS-ów takiego jak w GEMM.
  Aktualizacja blokowa rangi KC częściowo to poprawia (2*KC FLOP na element).
- **Widać wyraźny wpływ hierarchii pamięci**: dla SIZE ≤ 1000 macierz
  (≤ 8 MB) mieści się w L3 i wydajność wynosi 11–13 GFLOP/s; dla większych
  rozmiarów macierz przestaje się mieścić w L3 i wydajność spada do
  ~5–6 GFLOP/s, bo algorytm staje się ograniczony przepustowością RAM.
- **Wektoryzacja AVX2 + FMA daje największy pojedynczy zysk** – kernel
  przetwarza 4 double na instrukcję, a FMA łączy mnożenie z odejmowaniem,
  co w porównaniu do wersji skalarnej zmniejsza liczbę instrukcji ~8 razy.
- Wyniki obu wersji są **identyczne z dokładnością do błędów zaokrągleń**,
  co potwierdza poprawność optymalizacji.
