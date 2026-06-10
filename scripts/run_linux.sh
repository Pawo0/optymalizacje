#!/bin/bash
# Uruchom na maszynie docelowej (Linux) przed pomiarami / sprawozdaniem

echo "=== CPU ==="
lscpu 2>/dev/null | grep -E 'Model name|Architecture|CPU\(s\)|Thread|Core|MHz|cache|Flags' || \
    grep -m1 'model name' /proc/cpuinfo

echo ""
echo "=== Flagi SIMD (AVX2, FMA) ==="
grep -m1 flags /proc/cpuinfo | tr ' ' '\n' | grep -E '^(avx|avx2|fma|sse)' | sort -u

echo ""
echo "=== Cache ==="
lscpu 2>/dev/null | grep -i cache || getconf LEVEL1_DCACHE_SIZE 2>/dev/null

echo ""
echo "=== Kompilacja i test ==="
make native
./gauss_bench all
