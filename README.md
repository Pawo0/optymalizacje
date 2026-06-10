# Zadanie 2 — Eliminacja Gaussa (optymalizacja)

Implementacja eliminacji Gaussa z częściowym pivotingiem: wersja referencyjna
i zoptymalizowana (AVX2 + FMA). Program przeznaczony do uruchomienia na **Linuxie x86_64**.

## Szybki start (Linux)

```bash
# Na maszynie docelowej (serwer / lab):
make native          # kompilacja z -march=native
./gauss_bench all    # weryfikacja + benchmark

# Alternatywnie (gdy nie chcesz native):
make linux-avx2
./gauss_bench all
```

## Architektura projektu

```
include/gauss.h      — interfejs API
src/gauss_ref.c      — wersja referencyjna (skalarna)
src/gauss_opt.c      — wersja zoptymalizowana (AVX2 FMA)
src/verify.c         — testy poprawności
src/benchmark.c      — pomiary czasu i GFLOPS
src/main.c           — punkt wejścia
```

## Informacje o CPU (Linux)

Przed sprawozdaniem uruchom na maszynie docelowej:

```bash
lscpu | grep -E 'Model name|Architecture|Flags|cache'
# lub
cat /proc/cpuinfo | grep -E 'model name|flags|cache' | head -5
```

Szukaj flag: `avx2`, `fma`, `sse4_2`. Dla PAPI (opcjonalnie):

```bash
# Debian/Ubuntu
sudo apt install libpapi-dev
```

## GFLOPS

Liczba FLOP liczona analitycznie funkcją `gauss_flops(n)` — uwzględnia
eliminację w przód i podstawienie wstecz dla macierzy n×(n+1).

```
GFLOPS = gauss_flops(n) / czas_w_sekundach / 10^9
```

## macOS (development)

Na Macu (Apple Silicon) AVX2 nie jest dostępne — `make` buduje wersję skalarną
do testów poprawności. Pełne pomiary wydajności wykonaj na Linuxie.
