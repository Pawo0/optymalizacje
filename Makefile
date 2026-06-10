CC      = gcc
CFLAGS  = -std=c11 -Wall -O2 -Iinclude
LDFLAGS = -lm
TARGET  = gauss_bench

# na Linuxie x86: make AVX=1
ifeq ($(AVX),1)
CFLAGS += -mavx
endif

SRCS = src/main.c src/ge_ref.c src/ge_opt.c

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $(SRCS) $(LDFLAGS)

clean:
	rm -f $(TARGET) src/*.o

.PHONY: all clean
