#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "use: $0 file.c"
    echo "example: $0 ge8.c"
    exit 1
fi

SRC="$1"

if [ ! -f "$SRC" ]; then
    echo "Error, file '$SRC' doesn't exist."
    exit 1
fi

gcc -mavx -O2 "$SRC"

LAST_CHECK=""

for x in $(seq 100 100 2000); do
    OUTPUT=$(./a.out "$x")

    GFLOPS=$(echo "$OUTPUT" | awk -F ':' '/GFLOP\/s/ {
        gsub(/^[ \t]+/, "", $2)
        print $2
    }')

    CHECK=$(echo "$OUTPUT" | awk -F ':' '/Check/ {
        gsub(/^[ \t]+/, "", $2)
        print $2
    }')

    printf "%4d %s\n" "$x" "$GFLOPS"

    if [ "$x" -eq 1500 ]; then
        LAST_CHECK="$CHECK"
    fi
done

echo "Check: $LAST_CHECK"
