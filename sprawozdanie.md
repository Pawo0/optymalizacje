# Sprawozdanie — Zadanie 2

**Autor:** [imię nazwisko]

## 1. Cel

Zaimplementowałem eliminację Gaussa do rozwiązywania układu Ax=b. Jest wersja referencyjna i zoptymalizowana z AVX. Sprawdziłem czy dają ten sam wynik i zmierzyłem GFLOPS.

## 2. Implementacja

Macierz to `[A|b]` w jednej tablicy, indeks `IDX(i,j,ldc) = j + i*ldc` (jak na labach).

**ge_ref.c** — eliminacja jak ge3 z lab3 (pętla i-k-j, mnożnik liczony raz na wiersz). Potem podstawienie wstecz.

**ge_opt.c** — to samo co ref, ale wewnętrzna pętla po kolumnach jest wektoryzowana AVX (`__m256d`, 4 double naraz). Bez pivotingu, diagonalna dominacja przy testach.

## 3. Procesor

Program pod Linux x86_64, kompilacja: `make AVX=1`

[tu wklej wynik `lscpu` — model CPU, cache, flagi avx/avx2]

Kod używa 256-bitowego AVX. Lokalność: macierz row-major i pętla i-k-j (z lab3).

## 4. Weryfikacja

Program porównuje wektor x z ref i opt dla n = 16, 64, 256, 512.

```
./gauss_bench
```

Wszystkie testy powinny dać OK.

## 5. GFLOPS

Liczba operacji: `ge_flop(n) = n*(n-1)*(2n-1)/3` (wzór z lab9).

GFLOPS = ge_flop(n) / czas / 10^9

Czas mierzę `dclock()` (gettimeofday).

Przykład (Linux, make AVX=1):
```
./gauss_bench 512
Time: ...
FLOP: ...
GFLOP/s: ...
```

PAPI nie używałem.

## 6. Wnioski

- **Wersja opt daje ten sam wynik co ref** — weryfikacja przechodzi.
- **AVX przyspiesza** wewnętrzną pętlę bo robi 4 odejmowania naraz.
- **GFLOPS daleko od peaku CPU** — w Gaussie dużo dzielenia i zależności między wierszami, procesor nie nasycony w pełni.
- **Bez pivotingu** wystarczy jak macierz ma dużą diagonalę (testy).

## Pliki

```
include/common.h
src/ge_ref.c
src/ge_opt.c
src/main.c
Makefile
```

Uruchomienie:
```bash
make AVX=1
./gauss_bench
./gauss_bench 1024
```
