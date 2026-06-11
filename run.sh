#!/usr/bin/env bash

set -euo pipefail
export LC_ALL=C   # kropka dziesietna niezaleznie od locale

PEAK=64.0   # teoretyczny szczyt 1 rdzenia i5-8300H [GFLOP/s]

gcc -O2 -mavx2 -mfma ge.c -o ge

echo "Eliminacja Gaussa - Intel Core i5-8300H, gcc -O2 -mavx2 -mfma, 1 watek"
echo "Szczyt teoretyczny 1 rdzenia: ${PEAK} GFLOP/s (4.0 GHz x 2 FMA x 4 double x 2 op)"
echo ""
printf "%6s | %11s | %11s | %9s | %9s | %10s | %6s\n" \
       "SIZE" "ref [GF/s]" "opt [GF/s]" "speedup" "wysycenie" "max blad" "weryf"
printf -- "-------+-------------+-------------+-----------+-----------+------------+-------\n"

for x in $(seq 200 200 2000); do
    OUTPUT=$(./ge "$x")

    REF=$(echo "$OUTPUT" | awk -F ':' '/GFLOP\/s ref/ {gsub(/[ \t]/,"",$2); print $2}')
    OPT=$(echo "$OUTPUT" | awk -F ':' '/GFLOP\/s opt/ {gsub(/[ \t]/,"",$2); print $2}')
    ERR=$(echo "$OUTPUT" | awk -F ':' '/Max blad/ {gsub(/[ \t]/,"",$2); print $2}')
    WER=$(echo "$OUTPUT" | awk -F ':' '/Weryfikacja/ {gsub(/[ \t]/,"",$2); print $2}')

    SPEEDUP=$(awk -v r="$REF" -v o="$OPT" 'BEGIN {printf "%.1fx", o/r}')
    SAT=$(awk -v o="$OPT" -v p="$PEAK" 'BEGIN {printf "%.1f%%", 100.0*o/p}')

    printf "%6d | %11.2f | %11.2f | %9s | %9s | %10.1e | %6s\n" \
           "$x" "$REF" "$OPT" "$SPEEDUP" "$SAT" "$ERR" "$WER"
done
