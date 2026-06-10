#include "common.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

/* ge3 + wektoryzacja wewnetrznej petli (AVX) */
int ge_elim_opt(double *A, int n, int ldc)
{
  int i, j, k;
  double m;

  for (k = 0; k < n; k++) {
    for (i = k + 1; i < n; i++) {
      m = A[IDX(i, k, ldc)] / A[IDX(k, k, ldc)];

#if defined(__x86_64__) || defined(__i386__)
      j = k + 1;
      {
        const __m256d vm = _mm256_set1_pd(m);
        for (; j + 4 <= n + 1; j += 4) {
          __m256d ai = _mm256_loadu_pd(&A[IDX(i, j, ldc)]);
          __m256d ak = _mm256_loadu_pd(&A[IDX(k, j, ldc)]);
          ai = _mm256_sub_pd(ai, _mm256_mul_pd(ak, vm));
          _mm256_storeu_pd(&A[IDX(i, j, ldc)], ai);
        }
      }
      for (; j <= n; j++)
        A[IDX(i, j, ldc)] -= A[IDX(k, j, ldc)] * m;
#else
      for (j = k + 1; j <= n; j++)
        A[IDX(i, j, ldc)] -= A[IDX(k, j, ldc)] * m;
#endif
    }
  }
  return 0;
}

int ge_solve_opt(double *A, int n, int ldc, double *x)
{
  ge_elim_opt(A, n, ldc);
  ge_backsub(A, n, ldc, x);
  return 0;
}
