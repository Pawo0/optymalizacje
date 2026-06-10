#include "gauss.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef __linux__
#include <sys/time.h>
#endif

static double now_seconds(void) {
#ifdef __linux__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

typedef gauss_status_t (*gauss_fn)(matrix_t *m, double *x, size_t n);

static double benchmark_once(gauss_fn fn, size_t n, unsigned seed, int *status) {
    matrix_t m = matrix_alloc(n, n + 1);
    double *x = malloc(n * sizeof(double));

    matrix_fill_random(&m, seed);
    for (size_t i = 0; i < n; ++i) {
        m.data[i * m.cols + i] += (double)n;
    }

    const double t0 = now_seconds();
    const gauss_status_t s = fn(&m, x, n);
    const double t1 = now_seconds();

    *status = (int)s;
    matrix_free(&m);
    free(x);
    return t1 - t0;
}

static void run_case(const char *name, gauss_fn fn, size_t n, int repeats) {
    double best = 1e300;
    int status = GAUSS_OK;

    for (int r = 0; r < repeats; ++r) {
        int s = GAUSS_OK;
        const double t = benchmark_once(fn, n, (unsigned)(42 + r + n), &s);
        if (s != GAUSS_OK) {
            status = s;
            break;
        }
        if (t < best) {
            best = t;
        }
    }

    if (status != GAUSS_OK) {
        printf("  %-12s n=%4zu: BLAD (status=%d)\n", name, n, status);
        return;
    }

    const unsigned long long flops = gauss_flops(n);
    const double gflops = (double)flops / best / 1e9;

    printf("  %-12s n=%4zu: %8.4f ms  |  %llu FLOP  |  %8.3f GFLOPS\n",
           name, n, best * 1e3, flops, gflops);
}

int run_benchmark(void) {
    const size_t sizes[] = {64, 128, 256, 512, 1024};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    printf("=== Benchmark wydajnosci ===\n");
#if defined(__AVX2__) && defined(__FMA__)
    printf("SIMD: AVX2 + FMA wlaczone\n");
#elif defined(__AVX2__)
    printf("SIMD: AVX2 wlaczone\n");
#elif defined(__SSE2__)
    printf("SIMD: SSE2 (brak AVX2 w kompilacji)\n");
#else
    printf("SIMD: brak (skalar)\n");
#endif

    for (size_t t = 0; t < num_sizes; ++t) {
        const size_t n = sizes[t];
        const int repeats = (n >= 512) ? 3 : 5;
        printf("\nn = %zu\n", n);
        run_case("referencyjna", gauss_ref, n, repeats);
        run_case("zoptymaliz.", gauss_opt, n, repeats);
    }

    return 0;
}
