#!/usr/bin/env bash
# Run all 36 convergence cases (q_u = 1..6, two pairings, three fluxes) as
# independent Julia processes, JOBS at a time, then merge the results into
# results/convergence/results.csv.
#
#   JOBS=4 bash experiments/convergence/run_all.sh          # all degrees
#   QS="1 2 3" bash experiments/convergence/run_all.sh      # a subset
#
# The full sweep takes a few hours on a laptop; q_u = 6 dominates.
set -euo pipefail
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-4}
QS=${QS:-"6 5 4 3 2 1"}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-2}
D=results/convergence
mkdir -p "$D/parts" "$D/logs"
julia --project=. -e 'using Pkg; Pkg.instantiate(); using WesterveltEDG'

for q in $QS; do for p in matched mixed; do for f in central alternating upwind; do
  echo "$q $p $f"
done; done; done | xargs -P "$JOBS" -L 1 sh -c \
  'julia --project=. experiments/convergence/run_case.jl "$0" "$1" "$2" > results/convergence/logs/q$0_$1_$2.log 2>&1 && echo "finished q=$0 $1 $2"'

{ echo "q,pairing,flux,N,h,erru,errv,errux,errE,secs"
  for f in "$D"/parts/*.csv; do tail -n +2 "$f"; done; } > "$D/results.csv"
echo "wrote $D/results.csv ($(($(wc -l < "$D/results.csv") - 1)) rows)"
