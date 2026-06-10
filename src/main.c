#include "common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fill_matrix(double *A, int n, int ldc)
{
  srand(1);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= n; j++)
      A[IDX(i, j, ldc)] = (double)rand() / RAND_MAX;
    A[IDX(i, i, ldc)] += (double)n;
  }
}

static int verify(int n)
{
  const int ldc = n + 1;
  double *A1 = malloc((size_t)n * ldc * sizeof(double));
  double *A2 = malloc((size_t)n * ldc * sizeof(double));
  double *x1 = malloc((size_t)n * sizeof(double));
  double *x2 = malloc((size_t)n * sizeof(double));

  fill_matrix(A1, n, ldc);
  memcpy(A2, A1, (size_t)n * ldc * sizeof(double));

  ge_solve_ref(A1, n, ldc, x1);
  ge_solve_opt(A2, n, ldc, x2);

  int ok = 1;
  for (int i = 0; i < n; i++) {
    if (fabs(x1[i] - x2[i]) > 1e-9 * (fabs(x1[i]) + 1.0)) {
      ok = 0;
      break;
    }
  }

  free(A1);
  free(A2);
  free(x1);
  free(x2);
  return ok;
}

int main(int argc, char **argv)
{
  const int sizes[] = {16, 64, 256, 512};
  int bench_n = (argc > 1) ? atoi(argv[1]) : 512;

  printf("=== Weryfikacja ===\n");
  for (int i = 0; i < 4; i++) {
    int n = sizes[i];
    printf("n=%d: %s\n", n, verify(n) ? "OK" : "FAIL");
  }

  const int ldc = bench_n + 1;
  double *A = malloc((size_t)bench_n * ldc * sizeof(double));
  double *x = malloc((size_t)bench_n * sizeof(double));

  fill_matrix(A, bench_n, ldc);

  printf("\n=== Benchmark n=%d ===\n", bench_n);
  double t0 = dclock();
  ge_solve_opt(A, bench_n, ldc, x);
  double t1 = dclock();
  double time = t1 - t0;
  double flops = ge_flop(bench_n);

  printf("Time: %e s\n", time);
  printf("FLOP: %e\n", flops);
  printf("GFLOP/s: %e\n", flops / time / 1e9);

  free(A);
  free(x);
  return 0;
}
