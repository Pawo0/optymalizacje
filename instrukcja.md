# Instrukcja

## Pliki

- `ge.c` – cały kod: wersja referencyjna (`ge_ref`), zoptymalizowana
  (`ge_opt`), weryfikacja poprawności i pomiar czasu/GFLOPS.
- `run.sh` – kompiluje i odpala benchmark dla rozmiarów 200–2000.
- `sprawozdanie.md` – sprawozdanie (do konwersji na PDF przed wysłaniem na UPEL).

## Jak uruchomić

Pojedynczy rozmiar:

```bash
gcc -O2 -mavx2 -mfma ge.c -o ge
./ge 1000
```

Cały benchmark (sam kompiluje):

```bash
./run.sh
```

## Co program robi

Dla podanego rozmiaru N:

1. Tworzy dwie identyczne macierze N×N z losowymi wartościami (`srand(1)`,
   więc zawsze te same dane).
2. Na pierwszej wykonuje eliminację Gaussa wersją referencyjną (zwykła
   potrójna pętla), na drugiej wersją zoptymalizowaną.
3. Mierzy czas obu wersji i liczy GFLOP/s ze wzoru
   `FLOP = n(n-1)(2n-1)/3 + n(n-1)/2` (mnożenia + odejmowania + dzielenia).
4. Porównuje obie macierze wynikowe element po elemencie i wypisuje
   maksymalny błąd względny. Jak błąd < 1e-6 to wypisuje `Weryfikacja: OK`
   (małe różnice są normalne, bo zmieniona kolejność operacji
   zmiennoprzecinkowych daje inne zaokrąglenia).

## Jak działa optymalizacja (w skrócie)

- **AVX2 + FMA**: kernel aktualizuje po 8 elementów wiersza naraz dwoma
  rejestrami 256-bitowymi, operacja `c = c - p*m` to jedna instrukcja
  `fnmadd`. Twój i5-8300H ma AVX2 i FMA, stąd flagi `-mavx2 -mfma`.
- **Blokowanie KC=4**: macierz resztkowa jest aktualizowana raz na 4 kroki
  eliminacji naraz (`A22 = A22 - L21*U12`) zamiast po każdym kroku – mniej
  ruchu danych z pamięci.
- **Bloki MC=128, NC=256 + packing**: dane są przetwarzane kawałkami, które
  mieszczą się w cache L1/L2, i kopiowane do ciągłych buforów przed
  obliczeniami.

## Sprawozdanie do PDF

Na UPEL trzeba wrzucić PDF, np.:

```bash
pandoc sprawozdanie.md -o sprawozdanie.pdf
```

(albo otworzyć `sprawozdanie.md` w czymkolwiek i wydrukować do PDF).
