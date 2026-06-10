
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <immintrin.h>

#define BLKSIZE 8
#define IDX(i, j, n) ((j)+ (i)*(n))

#define MIN(a, b) ((a) < (b) ? (a) : (b))


#define MR 8
#define NR 8

#define MC 128
#define NC 256

#define KC 4

static double gtod_ref_time_sec = 0.0;

double ge_flop(int SIZE)
{
  double n = (double) SIZE;
  return n * (n - 1.0) * (2.0 * n - 1.0) / 3.0;
}

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

static void UpdateKernel(int m, int n, double *C,
                         int ldc, const double *pivot_row,
                         const double *mult)
{
  int i, j;

  for (j = 0; j + NR <= n; j += NR) {
    register __m256d p01 = _mm256_loadu_pd(pivot_row + j);
    register __m256d p45 = _mm256_loadu_pd(pivot_row + j + 4);

    for (i = 0; i < m; i++) {
      double *c = C + i * ldc + j;

      register __m256d mmult = _mm256_set1_pd(mult[i]);

      register __m256d c01 = _mm256_loadu_pd(c);
      register __m256d c45 = _mm256_loadu_pd(c + 4);

      c01 = _mm256_sub_pd(c01, _mm256_mul_pd(p01, mmult));
      c45 = _mm256_sub_pd(c45, _mm256_mul_pd(p45, mmult));

      _mm256_storeu_pd(c,     c01);
      _mm256_storeu_pd(c + 4, c45);
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

static void UpdateKernelK(int m, int n, int kdim, double *C,
                          int ldc, const double *U, const double *L)
{
  int i, j, r;

  for (j = 0; j + NR <= n; j += NR) {
    __m256d u01[KC];
    __m256d u45[KC];

    for (r = 0; r < kdim; r++) {
      u01[r] = _mm256_loadu_pd(U + r * n + j);
      u45[r] = _mm256_loadu_pd(U + r * n + j + 4);
    }

    for (i = 0; i < m; i++) {
      double *c = C + i * ldc + j;

      register __m256d c01 = _mm256_loadu_pd(c);
      register __m256d c45 = _mm256_loadu_pd(c + 4);

      for (r = 0; r < kdim; r++) {
        register __m256d mmult = _mm256_set1_pd(L[i * kdim + r]);

        c01 = _mm256_sub_pd(c01, _mm256_mul_pd(u01[r], mmult));
        c45 = _mm256_sub_pd(c45, _mm256_mul_pd(u45[r], mmult));
      }

      _mm256_storeu_pd(c,     c01);
      _mm256_storeu_pd(c + 4, c45);
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


int ge(double *A, int SIZE)
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

    /*
     * Faktoryzacja malego panelu k0 ... kend-1.
     *
     * Nie aktualizujemy od razu calej trailing matrix dla kazdego k.
     * Aktualizujemy tylko to, co jest konieczne, aby poprawnie wyznaczyc
     * kolejne pivoty i wiersze U wewnatrz panelu.
     */
    for (k = k0; k < kend; k++) {
      double pivot = A[IDX(k, k, SIZE)];

      /*
       * Aktualizacja kolumn panelu k+1 ... kend-1.
       * Te kolumny sa potrzebne do wyznaczenia kolejnych pivotow
       * oraz kolejnych mnoznikow w tym samym panelu.
       */
      for (jj = k + 1; jj < kend; jj++) {
        register double pj = A[IDX(k, jj, SIZE)];

        for (ii = k + 1; ii < SIZE; ii++) {
          A[IDX(ii, jj, SIZE)] = A[IDX(ii, jj, SIZE)]
          - pj * (A[IDX(ii, k, SIZE)] / pivot);
        }
      }

      /*
       * Wiersze lezace wewnatrz aktualnego panelu musza miec zaktualizowana
       * takze czesc po prawej stronie panelu. Beda one pozniej uzywane jako
       * wiersze U w aktualizacji blokowej.
       *
       * Wierszy ponizej panelu jeszcze tutaj nie aktualizujemy dla kolumn
       * kend ... SIZE-1. Ta praca zostanie wykonana pozniej przez
       * UpdateKernelK jako aktualizacja blokowa.
       */
      if (k + 1 < kend) {
        int ib = kend - (k + 1);

        for (jj = kend; jj < SIZE; jj += NC) {
          int jb = MIN(NC, SIZE - jj);

          PackPivotRow(jb, &A[IDX(k, jj, SIZE)], packed_pivot);
          PackMultipliers(ib, A, SIZE, k + 1, k, pivot, packed_mult_panel);

          UpdateKernel(ib,
                       jb,
                       &A[IDX(k + 1, jj, SIZE)],
                       SIZE,
                       packed_pivot,
                       packed_mult_panel);
        }
      }
    }

    /*
     * Aktualizacja trailing matrix:
     *
     *     A22 = A22 - L21 * U12
     *
     * To jest zasadnicza zmiana wzgledem ge9.c.
     * Zamiast wykonywac KC oddzielnych aktualizacji rangi 1,
     * wykonujemy jedna aktualizacje blokowa o szerokosci KC.
     */
    if (kend < SIZE) {
      PackLPanel(SIZE - kend, kb, A, SIZE, kend, k0, packed_l_all);

      for (jj = kend; jj < SIZE; jj += NC) {
        int jb = MIN(NC, SIZE - jj);

        PackUPanel(kb, jb, A, SIZE, k0, jj, packed_u);

        for (ii = kend; ii < SIZE; ii += MC) {
          int ib = MIN(MC, SIZE - ii);
          double *L = packed_l_all + (ii - kend) * kb;

          UpdateKernelK(ib,
                        jb,
                        kb,
                        &A[IDX(ii, jj, SIZE)],
                        SIZE,
                        packed_u,
                        L);
        }
      }
    }
  }

  free(packed_l_all);
  return 0;
}


int main( int argc, const char* argv[] )
{
  int i,j,k,iret;
  double dtime;
  int SIZE = atoi(argv[1]);
  double ** matrix;
  double *  matrix_;
  matrix_ = (double*) malloc(SIZE*SIZE*sizeof(double));
  matrix  = (double**) malloc(SIZE*sizeof(double*));
  
  for (i = 0; i < SIZE; i++) {
      matrix[i] = matrix_ + i*SIZE;
  }
  
  srand(1);
  for (i = 0; i < SIZE; i++) { 
    for (j = 0; j < SIZE; j++) { 
      matrix[i][j]=rand();
    }
  }
  printf("call GE");
  dtime = dclock();
  iret = ge(matrix_,SIZE);
  dtime = dclock()-dtime;
  printf("Time: %le \n", dtime);
  printf("FLOP: %le \n", ge_flop(SIZE));
  printf("GFLOP/s: %le \n", ge_flop(SIZE) / dtime / 1.0e9);

  double check=0.0;
  for (i = 0; i < SIZE; i++) {
    for (j = 0; j < SIZE; j++) {
      check = check + matrix[i][j];
    }
  }
  printf( "Check: %le \n", check);
  fflush( stdout );


  return iret;
}

