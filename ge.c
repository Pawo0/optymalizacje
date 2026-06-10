/*
 * Zadanie 2 - eliminacja Gaussa (bez pivotingu)
 *
 * ge_ref - wersja referencyjna (prosta potrojna petla)
 * ge_opt - wersja zoptymalizowana pod i5-8300H (Coffee Lake):
 *          AVX2 + FMA, blokowanie pod cache, packing danych
 *
 * Kompilacja: gcc -O2 -mavx2 -mfma ge.c -o ge
 * Uruchomienie: ./ge SIZE
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <immintrin.h>

#define IDX(i, j, n) ((j) + (i)*(n))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

/* parametry blokowania dobrane pod cache:
 * L1d = 32KB, L2 = 256KB, L3 = 8MB */
#define NR 8
#define MC 128
#define NC 256
#define KC 4

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */
double dclock()
{
  double the_time, norm_sec;
  struct timeval tv;
  gettimeofday( &tv, NULL );
  if ( gtod_ref_time_sec == 0.0 )
    gtod_ref_time_sec = ( double ) tv.tv_sec;
  norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;
  the_time = norm_sec + tv.tv_usec * 1.0e-6;
  return the_time;
}

/* rzeczywista liczba operacji zmiennoprzecinkowych:
 * dla kazdego k mamy (n-k-1) dzielen (mnozniki) oraz
 * 2*(n-k-1)^2 operacji (mnozenie + odejmowanie) w aktualizacji.
 * Po zsumowaniu:
 *   FLOP = n(n-1)(2n-1)/3 + n(n-1)/2
 */
double ge_flop(int SIZE)
{
  double n = (double) SIZE;
  return n * (n - 1.0) * (2.0 * n - 1.0) / 3.0 + n * (n - 1.0) / 2.0;
}

/* ------------------- wersja referencyjna ------------------- */

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

/* ------------------- wersja zoptymalizowana ------------------- */

static void PackPivotRow(int n, const double *src, double *dst)
{
  int j;
  for (j = 0; j < n; j++) {
    dst[j] = src[j];
  }
}

static void PackMultipliers(int m, const double *A, int SIZE, int row0, int k, double pivot, double *mult)
{
  int i;
  for (i = 0; i < m; i++) {
    mult[i] = A[IDX(row0 + i, k, SIZE)] / pivot;
  }
}

/* aktualizacja rangi 1: C = C - mult * pivot_row, AVX2 + FMA */
static void UpdateKernel(int m, int n, double *C,
                         int ldc, const double *pivot_row,
                         const double *mult)
{
  int i, j;

  for (j = 0; j + NR <= n; j += NR) {
    register __m256d p03 = _mm256_loadu_pd(pivot_row + j);
    register __m256d p47 = _mm256_loadu_pd(pivot_row + j + 4);

    for (i = 0; i < m; i++) {
      double *c = C + i * ldc + j;

      register __m256d mmult = _mm256_set1_pd(mult[i]);

      register __m256d c03 = _mm256_loadu_pd(c);
      register __m256d c47 = _mm256_loadu_pd(c + 4);

      /* fnmadd: c = c - p * m (jedna instrukcja FMA) */
      c03 = _mm256_fnmadd_pd(p03, mmult, c03);
      c47 = _mm256_fnmadd_pd(p47, mmult, c47);

      _mm256_storeu_pd(c,     c03);
      _mm256_storeu_pd(c + 4, c47);
    }
  }

  for (; j < n; j++) {
    register double pj = pivot_row[j];
    for (i = 0; i < m; i++) {
      C[i * ldc + j] = C[i * ldc + j] - pj * mult[i];
    }
  }
}

static void PackUPanel(int kdim, int n, const double *A, int SIZE,
                       int row0, int col0, double *U)
{
  int r, j;
  for (r = 0; r < kdim; r++) {
    for (j = 0; j < n; j++) {
      U[r * n + j] = A[IDX(row0 + r, col0 + j, SIZE)];
    }
  }
}

static void PackLPanel(int m, int kdim, const double *A, int SIZE,
                       int row0, int col0, double *L)
{
  int i, r;
  for (i = 0; i < m; i++) {
    for (r = 0; r < kdim; r++) {
      double pivot = A[IDX(col0 + r, col0 + r, SIZE)];
      L[i * kdim + r] = A[IDX(row0 + i, col0 + r, SIZE)] / pivot;
    }
  }
}

/* aktualizacja blokowa rangi KC: C = C - L * U, AVX2 + FMA */
static void UpdateKernelK(int m, int n, int kdim, double *C,
                          int ldc, const double *U, const double *L)
{
  int i, j, r;

  for (j = 0; j + NR <= n; j += NR) {
    __m256d u03[KC];
    __m256d u47[KC];

    for (r = 0; r < kdim; r++) {
      u03[r] = _mm256_loadu_pd(U + r * n + j);
      u47[r] = _mm256_loadu_pd(U + r * n + j + 4);
    }

    for (i = 0; i < m; i++) {
      double *c = C + i * ldc + j;

      register __m256d c03 = _mm256_loadu_pd(c);
      register __m256d c47 = _mm256_loadu_pd(c + 4);

      for (r = 0; r < kdim; r++) {
        register __m256d mmult = _mm256_set1_pd(L[i * kdim + r]);

        c03 = _mm256_fnmadd_pd(u03[r], mmult, c03);
        c47 = _mm256_fnmadd_pd(u47[r], mmult, c47);
      }

      _mm256_storeu_pd(c,     c03);
      _mm256_storeu_pd(c + 4, c47);
    }
  }

  for (; j < n; j++) {
    for (i = 0; i < m; i++) {
      register double cij = C[i * ldc + j];
      for (r = 0; r < kdim; r++) {
        cij = cij - U[r * n + j] * L[i * kdim + r];
      }
      C[i * ldc + j] = cij;
    }
  }
}

int ge_opt(double *A, int SIZE)
{
  int k0, k;
  int ii, jj;

  double packed_pivot[NC];
  double packed_mult_panel[KC];
  double packed_u[KC * NC];
  double *packed_l_all = (double*) malloc(SIZE * KC * sizeof(double));

  if (packed_l_all == NULL) {
    return -1;
  }

  for (k0 = 0; k0 < SIZE; k0 += KC) {
    int kb = MIN(KC, SIZE - k0);
    int kend = k0 + kb;

    /* faktoryzacja malego panelu k0 ... kend-1 */
    for (k = k0; k < kend; k++) {
      double pivot = A[IDX(k, k, SIZE)];

      /* aktualizacja kolumn wewnatrz panelu (potrzebne do kolejnych pivotow) */
      for (jj = k + 1; jj < kend; jj++) {
        register double pj = A[IDX(k, jj, SIZE)];
        for (ii = k + 1; ii < SIZE; ii++) {
          A[IDX(ii, jj, SIZE)] = A[IDX(ii, jj, SIZE)]
            - pj * (A[IDX(ii, k, SIZE)] / pivot);
        }
      }

      /* wiersze wewnatrz panelu musza byc zaktualizowane takze na prawo
       * od panelu - beda uzyte jako wiersze U w aktualizacji blokowej */
      if (k + 1 < kend) {
        int ib = kend - (k + 1);

        for (jj = kend; jj < SIZE; jj += NC) {
          int jb = MIN(NC, SIZE - jj);

          PackPivotRow(jb, &A[IDX(k, jj, SIZE)], packed_pivot);
          PackMultipliers(ib, A, SIZE, k + 1, k, pivot, packed_mult_panel);

          UpdateKernel(ib, jb, &A[IDX(k + 1, jj, SIZE)],
                       SIZE, packed_pivot, packed_mult_panel);
        }
      }
    }

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
  }

  free(packed_l_all);
  return 0;
}

/* ------------------- weryfikacja poprawnosci ------------------- */

/* maksymalny blad wzgledny pomiedzy dwoma macierzami */
double verify(const double *A, const double *B, int SIZE)
{
  int i, j;
  double max_err = 0.0;

  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      double a = A[IDX(i, j, SIZE)];
      double b = B[IDX(i, j, SIZE)];
      double err = fabs(a - b);
      if (fabs(a) > 1.0) {
        err = err / fabs(a);
      }
      if (err > max_err) {
        max_err = err;
      }
    }
  }
  return max_err;
}

void fill_matrix(double *A, int SIZE)
{
  int i, j;
  srand(1);
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      A[IDX(i, j, SIZE)] = rand();
    }
  }
}

int main( int argc, const char* argv[] )
{
  int iret;
  double dtime_ref, dtime_opt, err;
  int SIZE;
  double *A_ref, *A_opt;

  if (argc < 2) {
    printf("uzycie: %s SIZE\n", argv[0]);
    return 1;
  }
  SIZE = atoi(argv[1]);

  A_ref = (double *) malloc(SIZE*SIZE*sizeof(double));
  A_opt = (double *) malloc(SIZE*SIZE*sizeof(double));

  fill_matrix(A_ref, SIZE);
  fill_matrix(A_opt, SIZE);

  dtime_ref = dclock();
  iret = ge_ref(A_ref, SIZE);
  dtime_ref = dclock() - dtime_ref;

  dtime_opt = dclock();
  iret = ge_opt(A_opt, SIZE);
  dtime_opt = dclock() - dtime_opt;

  err = verify(A_ref, A_opt, SIZE);

  printf("SIZE: %d \n", SIZE);
  printf("FLOP: %le \n", ge_flop(SIZE));
  printf("Time ref: %le \n", dtime_ref);
  printf("Time opt: %le \n", dtime_opt);
  printf("GFLOP/s ref: %le \n", ge_flop(SIZE) / dtime_ref / 1.0e9);
  printf("GFLOP/s opt: %le \n", ge_flop(SIZE) / dtime_opt / 1.0e9);
  printf("Max blad wzgledny: %le \n", err);
  if (err < 1.0e-6) {
    printf("Weryfikacja: OK \n");
  } else {
    printf("Weryfikacja: BLAD \n");
  }
  fflush( stdout );

  free(A_ref);
  free(A_opt);

  return iret;
}
