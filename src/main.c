#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int run_verify(void);
int run_benchmark(void);

static void print_usage(const char *prog) {
    fprintf(stderr, "Uzycie: %s [verify|bench|all]\n", prog);
    fprintf(stderr, "  verify - testy poprawnosci\n");
    fprintf(stderr, "  bench  - pomiary wydajnosci i GFLOPS\n");
    fprintf(stderr, "  all    - oba (domyslnie)\n");
}

int main(int argc, char **argv) {
    const char *mode = "all";
    if (argc > 1) {
        mode = argv[1];
    }

    if (strcmp(mode, "verify") == 0) {
        return run_verify();
    }
    if (strcmp(mode, "bench") == 0) {
        return run_benchmark();
    }
    if (strcmp(mode, "all") == 0) {
        const int v = run_verify();
        printf("\n");
        run_benchmark();
        return v;
    }

    print_usage(argv[0]);
    return 1;
}
