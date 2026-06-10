#include "gauss.h"

#include <stdlib.h>
#include <string.h>

matrix_t matrix_alloc(size_t rows, size_t cols) {
    matrix_t m;
    m.rows = rows;
    m.cols = cols;
    m.data = calloc(rows * cols, sizeof(double));
    return m;
}

void matrix_free(matrix_t *m) {
    free(m->data);
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
}

static double pseudo_random(unsigned *state) {
    *state = *state * 1103515245u + 12345u;
    return (double)((*state >> 16) & 0x7FFF) / 32768.0;
}

void matrix_fill_random(matrix_t *m, unsigned seed) {
    unsigned state = seed ? seed : 1u;
    for (size_t i = 0; i < m->rows; ++i) {
        for (size_t j = 0; j < m->cols; ++j) {
            m->data[i * m->cols + j] = pseudo_random(&state) * 2.0 - 1.0;
        }
    }
}

void matrix_copy(const matrix_t *src, matrix_t *dst) {
    if (src->rows != dst->rows || src->cols != dst->cols) {
        return;
    }
    memcpy(dst->data, src->data, src->rows * src->cols * sizeof(double));
}
