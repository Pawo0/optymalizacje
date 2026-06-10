#include "common.h"

/* ge3 z lab3 — petla i-k-j */
int ge_elim_ref(double *A, int n, int ldc)
{
  int i, j, k;
  double m;

  for (k = 0; k < n; k++) {
    for (i = k + 1; i < n; i++) {
      m = A[IDX(i, k, ldc)] / A[IDX(k, k, ldc)];
      for (j = k + 1; j <= n; j++)
        A[IDX(i, j, ldc)] -= A[IDX(k, j, ldc)] * m;
    }
  }
  return 0;
}

void ge_backsub(const double *A, int n, int ldc, double *x)
{
  for (int i = n - 1; i >= 0; i--) {
    double s = A[IDX(i, n, ldc)];
    for (int j = i + 1; j < n; j++)
      s -= A[IDX(i, j, ldc)] * x[j];
    x[i] = s / A[IDX(i, i, ldc)];
  }
}

int ge_solve_ref(double *A, int n, int ldc, double *x)
{
  ge_elim_ref(A, n, ldc);
  ge_backsub(A, n, ldc, x);
  return 0;
}
