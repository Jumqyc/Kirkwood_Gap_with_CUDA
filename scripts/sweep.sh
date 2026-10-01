#!/usr/bin/env bash
#
# Fills a block of free time with a queue of runs, longest and most valuable
# first, so that whatever the budget turns out to be, the part that got done is
# the part worth having. Each run is delegated to run_one.sh, which brings its
# own resume, watchdog and restart.
#
#   nohup bash scripts/sweep.sh > data/sweep/driver.out 2>&1 &
#
# Default budget is 8 hours. Override with BUDGET_MINUTES.
#
# The queue:
#
#   e010_long  e_max=0.10  for 3 Myr. The 1 Myr epochs inside it are the old
#              sweep's reference point, and the extra time is what asks whether
#              the 3:1 resonance is simply slower than the 2:1 rather than
#              absent.
#   e005/e020/e030   e_max at 1 Myr. Where the 2:1 gap stops opening, at five
#              times the statistics of the sweep that posed the question.
#   e010sat    e_max=0.10 with Saturn, 1 Myr. Saturn made no measurable
#              difference at 20000 particles; this puts a bound on that.
#
# A run that is already complete is skipped, so re-running this after a reboot
# picks up where it left off.
set -euo pipefail

cd "$(dirname "$0")/.."

SWEEP_DIR="${SWEEP_DIR:-data/sweep}"
BUDGET_MINUTES="${BUDGET_MINUTES:-480}"
RUNNER="${RUNNER:-scripts/run_one.sh}"
BIN="${BIN:-build-o3/kirkwood_gpu}"

# name        N        years      e_max  saturn
QUEUE_DEFAULT=(
    "e010_long  100000   3000000    0.10   0"
    "e005       100000   1000000    0.05   0"
    "e020       100000   1000000    0.20   0"
    "e030       100000   1000000    0.30   0"
    "e010sat    100000   1000000    0.10   1"
)

# The queue can be replaced from the environment, entries separated by ';', so
# the script can be exercised at a scale that finishes in seconds:
#   QUEUE="tiny 20000 20 0.1 0; tinysat 20000 20 0.1 1" bash scripts/sweep.sh
if [ -n "${QUEUE:-}" ]; then
    IFS=';' read -r -a CONFIGS <<< "$QUEUE"
else
    CONFIGS=("${QUEUE_DEFAULT[@]}")
fi

# Measured ns per particle per step on this machine, from the N sweep. The
# nearest measured count is used for a configuration that is not itself in the
# table, which is accurate enough to decide what fits in the budget.
RATE_TABLE="20000:1.97 50000:1.32 100000:1.05 200000:0.87 500000:0.88 1000000:0.828"

rate_for() {
    awk -v n="$1" -v tbl="$RATE_TABLE" 'BEGIN{
        m = split(tbl, a, " ");
        best = 1e18; out = 0.828;
        for (i = 1; i <= m; i++) {
            split(a[i], p, ":");
            d = (p[1] > n ? p[1] - n : n - p[1]);
            if (d < best) { best = d; out = p[2] + 0; }
        }
        print out;
    }'
}

est_minutes() {
    awk -v n="$2" -v y="$3" -v r="$1" \
        'BEGIN{printf "%d", y * 36.525 * n * r / 1e9 / 60 * 1.1}'
}

[ -x "$BIN" ] || { echo "no binary at $BIN -- build first" >&2; exit 1; }
[ -f "$RUNNER" ] || { echo "no runner at $RUNNER" >&2; exit 1; }

echo "sweep        $SWEEP_DIR"
echo "budget       ${BUDGET_MINUTES} min"
echo "queue"
for cfg in "${CONFIGS[@]}"; do
    read -r name n years e_max sat <<< "$cfg"
    printf "  %-11s N=%-8s %-8s yr  e_max=%-5s saturn=%s  ~%s min\n" \
        "$name" "$n" "$years" "$e_max" "$sat" "$(est_minutes "$(rate_for "$n")" "$n" "$years")"
done
echo

# The free time belongs to this queue, so take the GPU rather than sharing it:
# two simulations on one card each run at about half speed, and the epochs a
# stopped run already wrote are kept, so nothing is lost by stopping it.
if pgrep -x kirkwood_gpu >/dev/null 2>&1 || pgrep -f "bash .*run_one.sh" >/dev/null 2>&1; then
    echo "stopping the run already on the GPU (its epochs stay on disk)"
    pkill -f "bash .*run_one.sh" 2>/dev/null || true
    sleep 2
    pkill -x kirkwood_gpu 2>/dev/null || true
    sleep 3
    pkill -9 -x kirkwood_gpu 2>/dev/null || true
    sleep 1
fi

start=$(date +%s)
ran=0
skipped=0
out_of_time=0

for cfg in "${CONFIGS[@]}"; do
    read -r name n years e_max sat <<< "$cfg"
    out="$SWEEP_DIR/$name"
    steps=$(awk -v y="$years" 'BEGIN{printf "%d", y * 36.525}')
    target="$out/epoch_${steps}.bin"

    if [ -f "$target" ]; then
        echo "[$name] already complete, skipping"
        skipped=$((skipped + 1))
        continue
    fi

    elapsed_min=$(( ($(date +%s) - start) / 60 ))
    left_min=$((BUDGET_MINUTES - elapsed_min))
    need_min=$(est_minutes "$(rate_for "$n")" "$n" "$years")

    # The estimate is pessimistic (rate table plus 10%), so a run that only just
    # fails to fit is worth starting anyway -- it resumes, and finishing part of
    # it is not wasted. Only skip when it clearly will not fit.
    if [ "$ran" -gt 0 ] && [ "$need_min" -gt $((left_min * 2)) ]; then
        echo "[$name] needs ~${need_min} min, ${left_min} min left; stopping the queue"
        out_of_time=1
        break
    fi

    echo
    echo "=== [$name] N=$n ${years} yr e_max=$e_max saturn=$sat  (~${need_min} min) ==="
    echo "    started $(date '+%H:%M:%S'), ${left_min} min of budget left"

    OUT="$out" N="$n" YEARS="$years" E_MAX="$e_max" SATURN="$sat" \
        bash "$RUNNER" || {
            echo "[$name] the runner gave up; continuing with the next one" >&2
            continue
        }
    ran=$((ran + 1))
done

total_min=$(( ($(date +%s) - start) / 60 ))
echo
echo "=== sweep finished at $(date '+%H:%M:%S') after ${total_min} min ==="
echo "    ran $ran, skipped $skipped$( [ "$out_of_time" -eq 1 ] && echo ', ran out of budget' )"
echo
for cfg in "${CONFIGS[@]}"; do
    read -r name n years e_max sat <<< "$cfg"
    out="$SWEEP_DIR/$name"
    steps=$(awk -v y="$years" 'BEGIN{printf "%d", y * 36.525}')
    if [ -f "$out/epoch_${steps}.bin" ]; then
        state="complete"
    else
        newest=$(ls -t "$out"/epoch_*.bin 2>/dev/null | head -1 || true)
        if [ -n "$newest" ]; then
            s=$(basename "$newest" .bin); s=${s#epoch_}
            state="$(awk -v s="$s" -v t="$steps" 'BEGIN{printf "%.1f%%", 100*s/t}')"
        else
            state="not started"
        fi
    fi
    printf "  %-11s %-10s %s\n" "$name" "$state" "$(du -sh "$out" 2>/dev/null | cut -f1)"
done
