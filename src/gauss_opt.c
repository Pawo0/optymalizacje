#include "gauss.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#define GAUSS_X86 1
#endif

#include <math.h>
#include <stddef.h>

static inline double *row_ptr(matrix_t *m, size_t r) {
    return m->data + r * m->cols;
}

/* Aktualizacja wiersza: row[j] -= factor * pivot[j] dla j in [start, cols) */
static void axpy_neg_row(double *row, const double *pivot, double factor,
                         size_t start, size_t cols) {
#if defined(GAUSS_X86) && defined(__AVX2__) && defined(__FMA__)
    const __m256d vf = _mm256_set1_pd(factor);
    size_t j = start;

    /* Wyrównanie do granicy 4 elementów AVX2 */
    const size_t aligned_start = (start + 3) & ~(size_t)3;
    for (; j < aligned_start && j < cols; ++j) {
        row[j] -= factor * pivot[j];
    }

    for (; j + 4 <= cols; j += 4) {
        __m256d vr = _mm256_loadu_pd(row + j);
        __m256d vp = _mm256_loadu_pd(pivot + j);
        vr = _mm256_fnmadd_pd(vf, vp, vr); /* row -= factor * pivot */
        _mm256_storeu_pd(row + j, vr);
    }

    for (; j < cols; ++j) {
        row[j] -= factor * pivot[j];
    }
#else
    for (size_t j = start; j < cols; ++j) {
        row[j] -= factor * pivot[j];
    }
#endif
}

gauss_status_t gauss_opt(matrix_t *m, double *x, size_t n) {
    if (!m || !x || m->rows < n || m->cols < n + 1) {
        return GAUSS_INVALID;
    }

    const size_t cols = m->cols;

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
            size_t k = i;
#if defined(GAUSS_X86) && defined(__AVX2__)
            for (; k + 4 <= cols; k += 4) {
                __m256d vi = _mm256_loadu_pd(row_i + k);
                __m256d vp = _mm256_loadu_pd(row_p + k);
                _mm256_storeu_pd(row_i + k, vp);
                _mm256_storeu_pd(row_p + k, vi);
            }
#endif
            for (; k < cols; ++k) {
                const double tmp = row_i[k];
                row_i[k] = row_p[k];
                row_p[k] = tmp;
            }
        }

        const double *pivot_row = row_ptr(m, i);
        const double pivot_val = pivot_row[i];

        for (size_t j = i + 1; j < n; ++j) {
            double *row_j = row_ptr(m, j);
            const double factor = row_j[i] / pivot_val;
            row_j[i] = 0.0;
            axpy_neg_row(row_j, pivot_row, factor, i + 1, cols);
        }
    }

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
