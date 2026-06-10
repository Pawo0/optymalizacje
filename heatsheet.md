# Jak działa program i jak go prezentować

## 1. Ogólny przebieg programu (od startu do końca)

Po uruchomieniu `./ge 1000` program robi po kolei:

1. Tworzy **dwie identyczne macierze** 1000×1000 wypełnione tymi samymi losowymi liczbami (bo `srand(1)` daje zawsze ten sam ciąg "losowych" liczb).
2. Na pierwszej macierzy odpala **wersję wolną** (`ge_ref`), na drugiej **wersję szybką** (`ge_opt`) i mierzy czas każdej.
3. **Porównuje obie macierze wynikowe** element po elemencie — jeśli wyniki są praktycznie takie same, optymalizacja niczego nie zepsuła.
4. Wypisuje czasy, GFLOP/s (ile miliardów operacji na sekundę procesor faktycznie wykonał) i wynik weryfikacji.

## 2. Co w ogóle robi eliminacja Gaussa

To algorytm, który zeruje elementy pod przekątną macierzy (sprowadza układ równań do postaci trójkątnej). Dla każdego kroku `k`:
- bierzemy wiersz `k` (tzw. **wiersz pivota**),
- od każdego wiersza poniżej odejmujemy ten wiersz pomnożony przez odpowiedni **mnożnik** (`A[i][k] / A[k][k]`), tak żeby w kolumnie `k` zrobiło się zero.

Wersja referencyjna to dosłownie ten przepis:

```57:70:ge.c
int ge_ref(double *A, int SIZE)
{
  int i, j, k;
  double multipler;
  for (k = 0; k < SIZE; k++) {
    for (i = k+1; i < SIZE; i++) {
      multipler = A[IDX(i, k, SIZE)] / A[IDX(k, k, SIZE)];
      for (j = k+1; j < SIZE; j++) {
        A[IDX(i, j, SIZE)] = A[IDX(i, j, SIZE)] - A[IDX(k, j, SIZE)] * multipler;
      }
    }
  }
  return 0;
}
```

**Dlaczego to jest wolne?** Procesor liczy tu jedną liczbę naraz i dla każdej policzonej liczby musi czekać na dane z pamięci. Procesor potrafi liczyć dużo szybciej, niż pamięć potrafi dostarczać dane — i to jest cały problem, który rozwiązują optymalizacje.

## 3. Trzy optymalizacje — gdzie są i jak działają

### Optymalizacja 1: Wektoryzacja AVX2 + FMA (liczymy 4 liczby naraz, dwie operacje w jednej)

**Gdzie:** w kernelach `UpdateKernel` i `UpdateKernelK`, np. tutaj:

```106:114:ge.c
      register __m256d c03 = _mm256_loadu_pd(c);
      register __m256d c47 = _mm256_loadu_pd(c + 4);

      /* fnmadd: c = c - p * m (jedna instrukcja FMA) */
      c03 = _mm256_fnmadd_pd(p03, mmult, c03);
      c47 = _mm256_fnmadd_pd(p47, mmult, c47);

      _mm256_storeu_pd(c,     c03);
      _mm256_storeu_pd(c + 4, c47);
```

**Jak to działa (na chłopski rozum):** Twój i5-8300H ma rejestry 256-bitowe (AVX2). Jedna liczba `double` ma 64 bity, więc do takiego rejestru wchodzą **4 liczby naraz** — i jedna instrukcja wykonuje działanie na wszystkich czterech jednocześnie. To jak myć 4 talerze jednym ruchem zamiast po jednym.

Do tego **FMA** (`_mm256_fnmadd_pd`): operacja `c = c - p*m` to normalnie dwie instrukcje (mnożenie, potem odejmowanie). FMA robi to **w jednej instrukcji**. Razem: 4 liczby × 2 operacje = 8 operacji w jednej instrukcji zamiast jednej.

Kernel przetwarza 8 elementów wiersza naraz (dwa rejestry po 4), a "ogon", który nie dzieli się przez 8, doliczany jest zwykłą pętlą skalarną (linie 118–123).

**Dopasowanie do procesora:** sprawdziłem flagi w `/proc/cpuinfo` — i5-8300H ma `avx2` i `fma`, stąd flagi kompilacji `-mavx2 -mfma`. Starszy procesor bez AVX2 by się na tym wywalił (illegal instruction).

### Optymalizacja 2: Aktualizacja blokowa rangi KC=4 (mniej kursów do pamięci)

**Gdzie:** cała struktura `ge_opt` — pętla po `k0` skacze co `KC=4` kroki, a sedno jest tu:

```241:258:ge.c
    /* aktualizacja blokowa trailing matrix: A22 = A22 - L21 * U12 */
    if (kend < SIZE) {
      PackLPanel(SIZE - kend, kb, A, SIZE, kend, k0, packed_l_all);

      for (jj = kend; jj < SIZE; jj += NC) {
        int jb = MIN(NC, SIZE - jj);

        PackUPanel(kb, jb, A, SIZE, k0, jj, packed_u);

        for (ii = kend; ii < SIZE; ii += MC) {
          int ib = MIN(MC, SIZE - ii);
          double *L = packed_l_all + (ii - kend) * kb;

          UpdateKernelK(ib, jb, kb, &A[IDX(ii, jj, SIZE)],
                        SIZE, packed_u, L);
        }
      }
    }
```

**Jak to działa:** w zwykłej wersji każdy krok `k` przechodzi przez **całą** pozostałą macierz — czyli dla macierzy 2000×2000 cała macierz (32 MB!) jest przepychana przez procesor 2000 razy. To jakby rozpakowywać i pakować całą szafę po to, żeby zmienić jedną koszulę.

W wersji blokowej najpierw robimy eliminację tylko na wąskim "panelu" 4 kolumn (linie 212–239), a potem aktualizujemy resztę macierzy **raz za wszystkie 4 kroki naraz** (`A22 = A22 - L21·U12` — to w praktyce małe mnożenie macierzy, dokładnie jak w instruktażu how-to-optimize-gemm). Efekt: każdy element macierzy jest ładowany z pamięci **raz na 4 kroki zamiast co krok**, a na każde załadowanie przypada 4× więcej obliczeń (wyższa "intensywność arytmetyczna").

W `UpdateKernelK` widać, że dla jednego załadowania `c03`/`c47` wykonujemy pętlę po `r` — czyli 4 operacje FMA, zanim wynik wróci do pamięci (linie 164–179).

### Optymalizacja 3: Blokowanie pod cache + packing (dane zawsze "pod ręką")

**Gdzie:** stałe na górze pliku i wywołania `Pack*`:

```21:26:ge.c
/* parametry blokowania dobrane pod cache:
 * L1d = 32KB, L2 = 256KB, L3 = 8MB */
#define NR 8
#define MC 128
#define NC 256
#define KC 4
```

**Jak to działa:** procesor ma małe, ale błyskawiczne pamięci podręczne (cache): L1 (32 KB), L2 (256 KB), L3 (8 MB). Dostęp do L1 trwa ~4 cykle, do RAM ~200+ cykli. Cache to jak biurko — masz na nim mało miejsca, ale wszystko w zasięgu ręki; RAM to piwnica.

Dlatego aktualizacja jest cięta na bloki **MC×NC = 128×256**: panel U (4×256 elementów = 8 KB) mieści się w L1, blok roboczy w L2. Procesor "miele" jeden blok do końca, zanim weźmie kolejny — dane nie wypadają z cache w połowie roboty.

**Packing** (`PackPivotRow`, `PackMultipliers`, `PackUPanel`, `PackLPanel`) to kopiowanie potrzebnych danych do małych, **ciągłych** buforów przed obliczeniem. Mnożniki leżą w macierzy w kolumnie, czyli co `SIZE*8` bajtów od siebie — fatalnie dla cache. Po skopiowaniu do bufora `mult[]` leżą obok siebie i czytają się sekwencyjnie.

## 4. Weryfikacja i GFLOPS — jak to liczę

- **Weryfikacja** (`verify`, linie 268–287): porównuję każdy element wyniku `ge_ref` i `ge_opt`, biorę maksymalny **błąd względny**. Jeśli < 1e-6 → OK. Błędy rzędu 1e-11–1e-7 są **normalne**: blokowanie i FMA zmieniają kolejność działań zmiennoprzecinkowych, a `(a+b)+c ≠ a+(b+c)` w arytmetyce float — to nie błąd algorytmu, tylko inne zaokrąglenia.
- **FLOP** (linie 43–53): zliczam rzeczywiste operacje — dla każdego `k`: `(n-k-1)` dzieleń + `2(n-k-1)²` mnożeń/odejmowań, co po zsumowaniu daje `n(n-1)(2n-1)/3 + n(n-1)/2 ≈ ⅔n³`. GFLOP/s = FLOP / czas / 10⁹.

## 5. Na czym się skupić przy prezentowaniu

Z tabeli ocen wynika, że prowadzący będzie pytał głównie o **zrozumienie mechanizmów**, nie o kod linijka po linijce. Kolejność ważności:

1. **Dopasowanie do TWOJEGO procesora** (to jest warunek zaliczenia!). Umiej powiedzieć z głowy: *i5-8300H, Coffee Lake, 4 rdzenie/8 wątków, AVX2 + FMA (256-bit, 4 double na rejestr), L1d 32 KB, L2 256 KB, L3 8 MB*. I że dlatego: `-mavx2 -mfma`, `fnmadd`, bloki 8 KB pod L1.

2. **Wykres/tabela wyników i ich INTERPRETACJA** — to robi największe wrażenie:
   - opt ~12.8 GFLOP/s vs ref ~4.5 → **przyspieszenie ~3×**;
   - **załamanie przy SIZE > 1000**: macierz 1000² double = 8 MB = dokładnie rozmiar L3. Większa macierz nie mieści się w cache i program zaczyna być ograniczony przepustowością RAM (spadek do ~5.5 GFLOP/s). Umiesz to policzyć na palcach — to jest złoto na obronie;
   - dlaczego nie 64 GFLOP/s (teoretyczny szczyt = 4 GHz × 2 jednostki FMA × 4 double × 2 op): bo eliminacja Gaussa ma **mało obliczeń na bajt danych** (2 FLOP na element przy randze 1, 8 przy KC=4) — w przeciwieństwie do GEMM nie da się jej w pełni "wykarmić" danymi.

3. **Trzy optymalizacje jednym zdaniem każda** (jeśli to umiesz, obronisz wszystko):
   - *SIMD+FMA*: "liczę 4 doubl'e jedną instrukcją, a mnożenie z odejmowaniem łączę w jedną operację fnmadd";
   - *blokowanie KC*: "aktualizuję resztę macierzy raz na 4 kroki eliminacji zamiast po każdym — 4× mniej ruchu z pamięci, jak w blocked LU/GEMM z instruktażu";
   - *cache + packing*: "tnę robotę na bloki mieszczące się w L1/L2 i kopiuję dane do ciągłych buforów, żeby czytały się sekwencyjnie".

4. **Weryfikacja**: umiej wyjaśnić, czemu błąd nie jest zerowy (kolejność operacji float) i czemu to jest OK.

Typowe pytania, na które warto być gotowym:
- *"Czemu nie używasz pivotingu?"* → zadanie dopuszcza wersję bez ("z lub bez pivotingu"); dla losowych macierzy pivot praktycznie nigdy nie jest zerem; pivoting psułby porównanie wydajności.
- *"Czemu KC=4, MC=128, NC=256?"* → dobrane pod hierarchię cache: panel U (KC×NC×8 B = 8 KB) mieści się w L1 (32 KB), blok MC×NC (256 KB) w L2/L3; KC większe = lepsza intensywność arytmetyczna, ale większe bufory — 4 to kompromis sprawdzony na laboratoriach.
- *"Co to jest `register __m256d`?"* → zmienna trzymana w 256-bitowym rejestrze wektorowym YMM, na której działają instrukcje AVX.
- *"Czemu pomiar jest jednowątkowy?"* → mierzymy efekt optymalizacji niskopoziomowych (SIMD, cache), wielowątkowość to osobny temat; GFLOP/s odnosimy do szczytu jednego rdzenia.

**Rada na sam występ:** zacznij od wyniku ("3× przyspieszenie, weryfikacja OK"), potem pokaż tabelę i wytłumacz załamanie przy 1000 przez rozmiar L3 — to jeden konkret, który od razu pokazuje, że rozumiesz, co się dzieje w sprzęcie, a nie tylko przepisałeś kod.