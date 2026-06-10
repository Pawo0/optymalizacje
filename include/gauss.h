#ifndef GAUSS_H
#define GAUSS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    GAUSS_OK = 0,
    GAUSS_SINGULAR = 1,
    GAUSS_INVALID = 2
} gauss_status_t;

/* Macierz w układzie wierszowym (row-major): data[i * cols + j] */
typedef struct {
    double *data;
    size_t rows;
    size_t cols;
} matrix_t;

matrix_t matrix_alloc(size_t rows, size_t cols);
void matrix_free(matrix_t *m);
void matrix_fill_random(matrix_t *m, unsigned seed);
void matrix_copy(const matrix_t *src, matrix_t *dst);

/* Rozwiązuje układ Ax = b; macierz m ma wymiar n x (n+1) = [A|b]. */
gauss_status_t gauss_ref(matrix_t *m, double *x, size_t n);
gauss_status_t gauss_opt(matrix_t *m, double *x, size_t n);

/* Liczba operacji zmiennoprzecinkowych dla eliminacji Gaussa (n x n). */
unsigned long long gauss_flops(size_t n);

#ifdef __cplusplus
}
#endif

#endif /* GAUSS_H */
