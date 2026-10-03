#!/usr/bin/env bash
#
# 1e6 test particles for 1e6 years: the run that asks whether the 3:1 resonance is
# simply slower than the 2:1, which 20 000 particles cannot
# answer. With a watchdog and automatic restart.
#
#   bash scripts/run_one.sh
#
# Expect about 8.5 hours and 3.1 GB of epoch files. Run it under nohup or tmux if
# the machine might sleep:
#
#   nohup bash scripts/run_one.sh > data/sweep/driver.out 2>&1 &
#
# Two things protect the run, and the first is useless without the second:
#
#   * The simulation resumes. If its output directory already holds epoch files
#     it continues from the newest one, so a restart costs at most one epoch
#     interval instead of the whole run.
#   * The watchdog. The GPU driver on WSL2 was observed hanging: the process
#     burns a core in a userspace spin, with no kernel running and no system
#     calls (0 context switches measured over 3 s), and cudaDeviceSynchronize
#     never returns. Here that shows up as "no new epoch for N seconds"; the
#     child is killed and the loop starts it again, which resumes.
#
# Everything can be overridden from the environment:
#   OUT=... N=... YEARS=... E_MAX=... SEED=... BIN=...
#   EPOCHS_TARGET=...  STALL_FACTOR=...  MAX_ATTEMPTS=...
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="${OUT:-data/run}"
N="${N:-1000000}"
YEARS="${YEARS:-1000000}"
E_MAX="${E_MAX:-0.1}"
SATURN="${SATURN:-0}"
ECCENTRIC="${ECCENTRIC:-0}"
SEED="${SEED:-114514}"
BIN="${BIN:-build-o3/kirkwood_gpu}"

# Dumps over the whole run. Each one is 48 MB at 1e6 particles, so 64 is about
# 3 GB and a couple of seconds of writing spread across the run.
EPOCHS_TARGET="${EPOCHS_TARGET:-64}"

# Kill the child once this many expected epoch durations pass with no new file.
# 2.5 leaves room for a loaded machine without letting a real hang sit for long.
STALL_FACTOR="${STALL_FACTOR:-2.5}"

# A hang is not the only way to fail, and a loop that restarts forever on a
# permanent error is worse than one that stops.
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"

# Run for at most this many minutes, then stop cleanly and leave the run
# resumable. 0 runs to completion in one go. Set it to work through a long run in
# slices -- each slice repeats at most one epoch interval of work, because the
# next invocation resumes from the newest epoch file.
#
#   SLICE_MINUTES=90 bash scripts/run_one.sh     # and again tomorrow
SLICE_MINUTES="${SLICE_MINUTES:-0}"

# dt is baked into the unit system in cpp/physics.hpp, so the year-to-step
# conversion has to be told what it is. Hardcoding 10 here silently produced a
# fifth of the requested time when dt was lowered to 2 for the convergence check.
DT_DAYS="${DT_DAYS:-10}"
STEPS=$(awk -v y="$YEARS" -v d="$DT_DAYS" 'BEGIN{printf "%d", y * 365.25 / d}')

EPOCH_EVERY=$(awk -v s="$STEPS" -v k="$EPOCHS_TARGET" \
    'BEGIN{e = int(s/k); printf "%d", (e > 0 ? e : 1)}')

EPOCHS=$(awk -v s="$STEPS" -v e="$EPOCH_EVERY" 'BEGIN{printf "%d", int(s/e)}')
DUMP_GB=$(awk -v n="$N" -v k="$EPOCHS" 'BEGIN{printf "%.1f", (k+1) * n * 48 / 1e9}')

# 0.8 ns per particle per step was measured at 1e6 on this machine; 0.9 is a
# deliberately pessimistic pace, so the stall threshold is not set too tight.
SEC_PER_EPOCH=$(awk -v n="$N" -v e="$EPOCH_EVERY" \
    'BEGIN{s = n * e * 0.9e-9; printf "%d", (s < 60 ? 60 : s)}')
STALL_SECONDS=$(awk -v s="$SEC_PER_EPOCH" -v f="$STALL_FACTOR" \
    'BEGIN{printf "%d", s * f}')

[ -x "$BIN" ] || { echo "no binary at $BIN -- build first" >&2; exit 1; }

mkdir -p "$OUT"
OUT_ABS=$(cd "$OUT" && pwd)
LOGFILE="$OUT_ABS/run.log"
TARGET="$OUT_ABS/epoch_${STEPS}.bin"

echo "n_particle   $N"
echo "years        $YEARS  ($STEPS steps of $DT_DAYS days)"
echo "e_max        $E_MAX"
echo "saturn       $SATURN"
echo "eccentric    $ECCENTRIC"
echo "seed         $SEED"
echo "epoch_every  $EPOCH_EVERY  (~$EPOCHS epochs, ~$DUMP_GB GB)"
echo "watchdog     kill after ${STALL_SECONDS}s with no new epoch (~${SEC_PER_EPOCH}s per epoch)"
echo "slice        ${SLICE_MINUTES} min per invocation (0 = run to completion)"
echo "out          $OUT_ABS"
echo "commit       $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo

interrupted=0
sim_pid=""

# The simulation is a child process, so anything that ends this script has to
# take the child with it -- otherwise an 8-hour run keeps going with nothing left
# to report it. A trap only runs once the foreground command returns, so a long
# sleep is a long window in which the shell is dead but the child is not; that is
# why the poll below is 5 s and prints only on an epoch change.
cleanup() { [ -n "$sim_pid" ] && kill "$sim_pid" 2>/dev/null || true; }
trap 'interrupted=1' INT TERM HUP
trap 'cleanup; exit 130' PIPE
trap cleanup EXIT

total_start=$(date +%s)
attempt=0
last_change=$(date +%s)
sliced=0
slice_start=$(date +%s)

printf "%8s %10s %12s %14s   %s\n" elapsed done remaining "ns/particle/step" state
while [ ! -f "$TARGET" ]; do
    attempt=$((attempt + 1))
    if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
        echo "gave up after $MAX_ATTEMPTS attempts" >&2
        exit 1
    fi
    [ "$interrupted" -eq 1 ] && exit 130

    [ "$attempt" -gt 1 ] && echo "restart #$((attempt - 1)) at $(date '+%H:%M:%S')"

    "$BIN" "$N" "$STEPS" "$OUT_ABS" "$EPOCH_EVERY" "$E_MAX" "$SATURN" "$SEED" \
        "$ECCENTRIC" \
        >> "$LOGFILE" 2>&1 &
    sim_pid=$!

    # The rate is measured from the newest epoch file's timestamp, not from the
    # wall clock: that is when those steps were actually finished, and using
    # "now" would fold in the poll interval and overstate the cost.
    last_state=$(ls -t "$OUT_ABS"/epoch_*.bin 2>/dev/null | head -1 || true)
    last_change=$(date +%s)

    while kill -0 "$sim_pid" 2>/dev/null; do
        sleep 5
        if [ "$interrupted" -eq 1 ]; then cleanup; exit 130; fi

        newest=$(ls -t "$OUT_ABS"/epoch_*.bin 2>/dev/null | head -1 || true)
        if [ "$newest" != "$last_state" ]; then
            last_state="$newest"
            last_change=$(date +%s)

            now=$(date +%s)
            elapsed=$((now - total_start))
            done_n=$(basename "$newest" .bin); done_n=${done_n#epoch_}
            t_done=$(( $(stat -c %Y "$newest") - total_start ))
            rate=$(awk -v e="$t_done" -v s="$done_n" -v n="$N" \
                'BEGIN{ if (s > 0 && e > 0) printf "%.3f", e*1e9/s/n; else printf "-" }')
            left=$(awk -v e="$t_done" -v s="$done_n" -v t="$STEPS" \
                'BEGIN{ if (s > 0 && e > 0) { r=(t-s)*e/s; printf "%d:%02d:%02d", r/3600, (r%3600)/60, r%60 } else printf "-" }')
            printf "%7dm %9.1f%% %12s %14s   %s\n" \
                $((elapsed / 60)) \
                "$(awk -v s="$done_n" -v t="$STEPS" 'BEGIN{printf "%.1f", 100*s/t}')" \
                "$left" "$rate" "$(basename "$newest")"
        fi

        if [ $(( $(date +%s) - last_change )) -gt "$STALL_SECONDS" ]; then
            echo "stalled: no new epoch for ${STALL_SECONDS}s; killing the child" >&2
            kill "$sim_pid" 2>/dev/null || true
            sleep 5
            kill -9 "$sim_pid" 2>/dev/null || true
            break
        fi

        # A slice stops the run between epochs, never inside one: the epoch file
        # on disk is the only state that survives, and an epoch is written whole
        # or not at all.
        if [ "$SLICE_MINUTES" -gt 0 ] &&
           [ $(( ($(date +%s) - slice_start) / 60 )) -ge "$SLICE_MINUTES" ]; then
            echo "slice of ${SLICE_MINUTES}m done at $(date '+%H:%M:%S');" \
                 "stopping. Re-run the same command to continue." >&2
            sliced=1
            cleanup
            wait "$sim_pid" 2>/dev/null || true
            sim_pid=""
            break 2
        fi
    done

    # Whether to restart is decided by how the child ended. A kill -- ours from
    # the watchdog, or anyone else's -- arrives as a signal, which bash reports
    # as 128 + N; the program's own error paths return a small status and must
    # stop the loop instead of being retried 20 times.
    status=0
    wait "$sim_pid" || status=$?
    sim_pid=""

    [ "$interrupted" -eq 1 ] && exit 130

    if [ "$status" -eq 0 ] && [ -f "$TARGET" ]; then
        continue # the while condition ends the loop
    fi
    if [ "$status" -ge 128 ]; then
        echo "the simulation was killed (status $status); restarting" >&2
        continue
    fi
    echo "the simulation exited with status $status before reaching step $STEPS" >&2
    tail -5 "$LOGFILE" >&2
    exit 1
done

total=$(( $(date +%s) - total_start ))
done_step=$(basename "$(ls -t "$OUT_ABS"/epoch_*.bin | head -1)" .bin)
done_step=${done_step#epoch_}
echo
if [ "$sliced" -eq 1 ]; then
    echo "slice finished: step ${done_step} of ${STEPS}" \
         "($(awk -v s="$done_step" -v t="$STEPS" 'BEGIN{printf "%.1f", 100*s/t}')%)"
    echo "this invocation ran $((total / 60))m; re-run the same command to continue"
else
    echo "reached step $STEPS in $((total / 3600))h $((total % 3600 / 60))m after $attempt attempt(s)"
fi
echo
grep -E "^backend=|^resuming" "$LOGFILE" | tail -4 || true
echo
echo "epoch files: $(ls "$OUT_ABS"/epoch_*.bin | wc -l), $(du -sh "$OUT_ABS" | cut -f1)"
echo
echo "next: point the analyser at $OUT_ABS"
