#!/usr/bin/env bash

set -euo pipefail

gcc -O2 -mavx2 -mfma ge.c -o ge

printf "%6s %12s %12s %16s %8s\n" "SIZE" "ref[GF/s]" "opt[GF/s]" "max_blad" "weryf"

for x in $(seq 200 200 2000); do
    OUTPUT=$(./ge "$x")

    REF=$(echo "$OUTPUT"  | awk -F ':' '/GFLOP\/s ref/ {gsub(/[ \t]/,"",$2); print $2}')
    OPT=$(echo "$OUTPUT"  | awk -F ':' '/GFLOP\/s opt/ {gsub(/[ \t]/,"",$2); print $2}')
    ERR=$(echo "$OUTPUT"  | awk -F ':' '/Max blad/ {gsub(/[ \t]/,"",$2); print $2}')
    WER=$(echo "$OUTPUT"  | awk -F ':' '/Weryfikacja/ {gsub(/[ \t]/,"",$2); print $2}')

    printf "%6d %12s %12s %16s %8s\n" "$x" "$REF" "$OPT" "$ERR" "$WER"
done
