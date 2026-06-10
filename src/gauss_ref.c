#include "gauss.h"

#include <math.h>
#include <stddef.h>

static inline double *row_ptr(matrix_t *m, size_t r) {
    return m->data + r * m->cols;
}

gauss_status_t gauss_ref(matrix_t *m, double *x, size_t n) {
    if (!m || !x || m->rows < n || m->cols < n + 1) {
        return GAUSS_INVALID;
    }

    const size_t cols = m->cols;

    /* Eliminacja w przód z częściowym wyborem elementu głównego */
    for (size_t i = 0; i < n; ++i) {
        size_t pivot = i;
        double max_val = fabs(row_ptr(m, i)[i]);

        for (size_t r = i + 1; r < n; ++r) {
            const double val = fabs(row_ptr(m, r)[i]);
            if (val > max_val) {
                max_val = val;
                pivot = r;
            }
        }

        if (max_val < 1e-14) {
            return GAUSS_SINGULAR;
        }

        if (pivot != i) {
            double *row_i = row_ptr(m, i);
            double *row_p = row_ptr(m, pivot);
            for (size_t k = i; k < cols; ++k) {
                const double tmp = row_i[k];
                row_i[k] = row_p[k];
                row_p[k] = tmp;
            }
        }

        const double pivot_val = row_ptr(m, i)[i];
        for (size_t j = i + 1; j < n; ++j) {
            double *row_j = row_ptr(m, j);
            const double factor = row_j[i] / pivot_val;
            row_j[i] = 0.0;

            for (size_t k = i + 1; k < cols; ++k) {
                row_j[k] -= factor * row_ptr(m, i)[k];
            }
        }
    }

    /* Podstawienie wstecz */
    for (size_t i = n; i-- > 0;) {
        double sum = row_ptr(m, i)[n];
        for (size_t j = i + 1; j < n; ++j) {
            sum -= row_ptr(m, i)[j] * x[j];
        }
        const double diag = row_ptr(m, i)[i];
        if (fabs(diag) < 1e-14) {
            return GAUSS_SINGULAR;
        }
        x[i] = sum / diag;
    }

    return GAUSS_OK;
}

unsigned long long gauss_flops(size_t n) {
    /*
     * Dla macierzy n x (n+1) [A|b]:
     * - eliminacja w przód: dla każdego i, (n-i-1) wierszy,
     *   każdy: 1 dzielenie + 2*(n-i) mnożeń/odejmowań (kolumny i..n)
     * - podstawienie wstecz: dla każdego i, (n-i-1) mnożeń + (n-i-1) odejmowań + 1 dzielenie
     */
    unsigned long long forward = 0;
    for (size_t i = 0; i + 1 < n; ++i) {
        const size_t rows = n - i - 1;
        const size_t cols = n - i; /* od i do n włącznie z b */
        forward += rows * (1ULL + 2ULL * cols);
    }

    unsigned long long backsub = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t k = n - i - 1;
        backsub += 2ULL * k + 1ULL;
    }

    return forward + backsub;
}
