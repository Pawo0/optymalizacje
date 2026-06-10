# Instrukcja — Zadanie 2 (eliminacja Gaussa)

## Jak uruchomić

### Linux (maszyna docelowa, x86)

```bash
cd zad2
make AVX=1          # kompilacja z wektoryzacja AVX
./gauss_bench       # weryfikacja + benchmark dla n=512
./gauss_bench 1024  # to samo, ale benchmark dla n=1024
```

Zeby wyczyscic skompilowany program:

```bash
make clean
```

### macOS

Na Macu (Apple Silicon) AVX nie dziala, ale program i tak sie zbuduje — optymalizacja wtedy robi to samo co ref (zwykla petla):

```bash
make
./gauss_bench
```

---

## Co zobaczysz po uruchomieniu

```
=== Weryfikacja ===
n=16: OK
n=64: OK
n=256: OK
n=512: OK

=== Benchmark n=512 ===
Time: 9.625000e-03 s      <- ile trwalo
FLOP: 8.921651e+07        <- ile operacji (wzor z lab9)
GFLOP/s: 9.269248e+00     <- FLOP / czas / 10^9
```

- **Weryfikacja** — sprawdza, czy wersja opt daje ten sam wynik co ref.
- **Benchmark** — mierzy czas wersji opt i liczy GFLOPS.

---

## Gdzie jest co — mapa plikow

```
zad2/
├── include/common.h    <- wspolne rzeczy (makro, czas, FLOP)
├── src/
│   ├── ge_ref.c        <- wersja referencyjna (prosta)
│   ├── ge_opt.c        <- wersja z AVX
│   └── main.c          <- start programu, testy, pomiar
├── Makefile            <- kompilacja
├── sprawozdanie.md     <- sprawozdanie do oddania
├── zadanie2.pdf        <- polecenie
└── prev/               <- stare laby (tylko do podgladu)
```

---

## Co robi kazdy plik

### `include/common.h`

- **`IDX(i, j, ldc)`** — liczy pozycje elementu w tablicy. Macierz jest w pamieci ciaglej, wiersz po wierszu.
- **`dclock()`** — mierzy czas (jak na labach, `gettimeofday`).
- **`ge_flop(n)`** — ile operacji zmiennoprzecinkowych robi eliminacja dla macierzy n×n (wzor z lab9).
- Deklaracje funkcji z `ge_ref.c` i `ge_opt.c`.

### `src/ge_ref.c` — wersja referencyjna

Trzy funkcje:

1. **`ge_elim_ref`** — eliminacja Gaussa w przod (jak ge3 z lab3)
2. **`ge_backsub`** — podstawienie wstecz, liczy wektor rozwiazania `x`
3. **`ge_solve_ref`** — wola oba kroki po kolei

### `src/ge_opt.c` — wersja zoptymalizowana

To samo co ref, ale w **`ge_elim_opt`** wewnetrzna petla po kolumnach uzywa **AVX** — 4 liczby `double` naraz zamiast jednej. Reszta identyczna.

### `src/main.c` — punkt wejscia

1. **`fill_matrix`** — losuje macierz `[A|b]`, dodaje `n` na diagonalę (zeby sie nie wysypalo bez pivotingu)
2. **`verify`** — kopiuje macierz, rozwiazuje ref i opt, porownuje `x`
3. **`main`** — odpala weryfikacje dla 4 rozmiarow, potem benchmark opt

---

## Jak to dziala — algorytm krok po kroku

Program rozwiazuje uklad **Ax = b**.

Macierz trzymasz jako jedna tablice **`[A | b]`** — n wierszy, n+1 kolumn (ostatnia to wektor b):

```
[ a11  a12  ...  a1n | b1 ]
[ a21  a22  ...  a2n | b2 ]
[  .    .   ...   .  | .  ]
[ an1  an2  ...  ann | bn ]
```

### Krok 1: Eliminacja w przod (`ge_elim_ref` / `ge_elim_opt`)

Dla kazdej kolumny `k` (element glowny):

1. Wez wiersz `k` jako wiersz pivot
2. Dla kazdego wiersza `i` ponizej:
   - policz mnoznik: `m = A[i][k] / A[k][k]`
   - odejmij `m × (wiersz k)` od wiersza `i` (zerujesz element pod diagonalą)

Po eliminacji masz macierz gorna trojkatna — nadal z kolumna `b` po prawej.

Petla jest w kolejnosci **i-k-j** (ge3) — mnoznik `m` liczony raz na wiersz, nie w kazdej iteracji po kolumnach.

### Krok 2: Podstawienie wstecz (`ge_backsub`)

Idziesz od ostatniego wiersza do gory:

- z ostatniego rownania masz od razu `x[n-1]`
- z przedostatniego wstawiasz juz znane `x` i liczysz kolejne
- itd. az do `x[0]`

### Roznica ref vs opt

| | ref | opt |
|---|-----|-----|
| Eliminacja | zwykla petla | ta sama petla, ale 4 elementy naraz (AVX) |
| Podstawienie wstecz | skalarne | to samo (wspolna funkcja w `ge_ref.c`) |
| Wynik | identyczny | identyczny (stad OK w weryfikacji) |

---

## Przeplyw programu

```
main()
  |
  +-- verify(16), verify(64), verify(256), verify(512)
  |     +-- ge_solve_ref -> x1
  |     +-- ge_solve_opt -> x2
  |     +-- porownaj x1 i x2
  |
  +-- benchmark: ge_solve_opt na macierzy n×(n+1)
        +-- dclock() -> ge_flop() -> GFLOPS
```

---

## Makefile w skrocie

- **`make`** — kompiluje `gauss_bench` z `-O2`
- **`make AVX=1`** — dodaje `-mavx` (potrzebne na Linuxie x86, zeby AVX dzialalo)
- **`make clean`** — usuwa skompilowany plik

Jesli cos nie dziala na Linuxie, sprawdz czy masz `gcc` i czy kompilujesz z `make AVX=1`.
