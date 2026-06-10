#include "gauss.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int compare_vectors(const double *a, const double *b, size_t n, double tol) {
    for (size_t i = 0; i < n; ++i) {
        const double diff = fabs(a[i] - b[i]);
        const double scale = fmax(fabs(a[i]), fmax(fabs(b[i]), 1.0));
        if (diff > tol * scale) {
            fprintf(stderr,
                    "  rozbieznosc w x[%zu]: ref=%.12e opt=%.12e diff=%.12e\n",
                    i, a[i], b[i], diff);
            return 0;
        }
    }
    return 1;
}

static int verify_size(size_t n, unsigned seed) {
    matrix_t ref_m = matrix_alloc(n, n + 1);
    matrix_t opt_m = matrix_alloc(n, n + 1);
    double *x_ref = malloc(n * sizeof(double));
    double *x_opt = malloc(n * sizeof(double));

    matrix_fill_random(&ref_m, seed);
    matrix_copy(&ref_m, &opt_m);

    /* Upewnij sie, ze macierz nie jest zdegenerowana: dominacja diagonalna */
    for (size_t i = 0; i < n; ++i) {
        ref_m.data[i * ref_m.cols + i] += (double)n;
        opt_m.data[i * opt_m.cols + i] = ref_m.data[i * ref_m.cols + i];
    }

    const gauss_status_t s_ref = gauss_ref(&ref_m, x_ref, n);
    const gauss_status_t s_opt = gauss_opt(&opt_m, x_opt, n);

    int ok = 1;
    if (s_ref != s_opt) {
        fprintf(stderr, "n=%zu: rozny status (ref=%d, opt=%d)\n", n, s_ref, s_opt);
        ok = 0;
    } else if (s_ref == GAUSS_OK &&
               !compare_vectors(x_ref, x_opt, n, 1e-10)) {
        ok = 0;
    }

    matrix_free(&ref_m);
    matrix_free(&opt_m);
    free(x_ref);
    free(x_opt);
    return ok;
}

int run_verify(void) {
    const size_t sizes[] = {4, 8, 16, 32, 64, 128, 256, 512};
    const size_t num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    int all_ok = 1;

    printf("=== Weryfikacja poprawnosci ===\n");
    for (size_t t = 0; t < num_sizes; ++t) {
        const size_t n = sizes[t];
        const int ok = verify_size(n, (unsigned)(1000 + n));
        printf("  n=%4zu: %s\n", n, ok ? "OK" : "FAIL");
        if (!ok) {
            all_ok = 0;
        }
    }

    printf("Wynik: %s\n", all_ok ? "WSZYSTKIE TESTY OK" : "BLAD");
    return all_ok ? 0 : 1;
}
