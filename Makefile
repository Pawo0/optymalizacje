# Makefile — cel: Linux x86_64 (AVX2/FMA)
# Na maszynie docelowej uzyj: make native   (march=native)
# Na macOS (dev): make          (bez AVX2, tylko test poprawnosci)

CC      := gcc
CFLAGS  := -std=c11 -Wall -Wextra -O3 -Iinclude
LDFLAGS := -lm

SRCS := src/main.c src/matrix.c src/gauss_ref.c src/gauss_opt.c \
        src/verify.c src/benchmark.c
OBJS := $(SRCS:.c=.o)
TARGET := gauss_bench

.PHONY: all native clean run verify bench

all: $(TARGET)

# Kompilacja pod Linux x86_64 z wektoryzacja AVX2 + FMA
linux-avx2: CFLAGS += -mavx2 -mfma
linux-avx2: clean $(TARGET)
	@echo "Zbudowano wersje AVX2+FMA (Linux x86_64)"

# Na maszynie docelowej — automatyczne dopasowanie do CPU
native: CFLAGS += -march=native
native: clean $(TARGET)
	@echo "Zbudowano z -march=native"

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET) all

verify: $(TARGET)
	./$(TARGET) verify

bench: $(TARGET)
	./$(TARGET) bench

clean:
	rm -f $(OBJS) $(TARGET)
