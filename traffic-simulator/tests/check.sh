#!/usr/bin/env bash
# All versions must print exactly the same hourly report as the sequential
# one, for several producer/consumer and rank counts, on the small sample and
# on a bigger generated file whose MPI messages are far over 4 KB.
set -euo pipefail
cd "$(dirname "$0")/.."
MPIRUN="${MPIRUN:-mpirun}"
MPI_EXTRA="${MPI_EXTRA:-}"
fail=0

body() { grep -v -e '^version:' -e '^seconds:' -e '^largest_result_message_bytes:'; }

mkdir -p data
[[ -f data/week.csv ]] || python3 tests/generate_data.py --days 7 --lights 25 --seed 3 --out data/week.csv

for file in data/sample.csv data/week.csv; do
    expected=$(./bin/sequential --file $file | body)
    check() {
        local name="$1"; shift
        if [[ "$("$@" | body)" == "$expected" ]]; then echo "PASS  $name"; else echo "FAIL  $name"; fail=1; fi
    }
    for pcb in "1 1 1" "4 1 1" "1 4 1" "3 2 1" "3 2 7" "2 2 100000"; do
        set -- $pcb
        check "threads p=$1 c=$2 batch=$3 $file" ./bin/threads --file $file --producers $1 --consumers $2 --batch $3 --queue 8
    done
    for r in 1 2 5; do
        check "mpi ranks=$r $file" $MPIRUN $MPI_EXTRA -n $r ./bin/mpi --file $file
    done
done

# The hourly report must be in time order across midnight.
first_hours=$(./bin/sequential --file data/sample.csv | head -10 | cut -c1-16)
if [[ "$first_hours" == "$(sort <<<"$first_hours")" ]]; then echo "PASS  hours in chronological order"; else echo "FAIL  hour order"; fail=1; fi

[[ $fail -eq 0 ]] && echo "All versions print the same report." || { echo "Differences found."; exit 1; }
