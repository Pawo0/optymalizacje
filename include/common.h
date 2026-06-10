#ifndef COMMON_H
#define COMMON_H

#include <sys/time.h>

#define IDX(i, j, ldc) ((j) + (i) * (ldc))

static double gtod_ref_time_sec = 0.0;

static inline double dclock(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  if (gtod_ref_time_sec == 0.0)
    gtod_ref_time_sec = (double)tv.tv_sec;
  return (double)tv.tv_sec - gtod_ref_time_sec + tv.tv_usec * 1.0e-6;
}

/* wzor z lab9 */
static inline double ge_flop(int n)
{
  double nd = (double)n;
  return nd * (nd - 1.0) * (2.0 * nd - 1.0) / 3.0;
}

void ge_backsub(const double *A, int n, int ldc, double *x);
int ge_solve_ref(double *A, int n, int ldc, double *x);
int ge_solve_opt(double *A, int n, int ldc, double *x);

#endif
