#!/bin/bash
# Rebuilds data/sweep after it was deleted.
#
# Seven configurations, all at 1e5 particles with seed 114514, six of them the
# ones scripts/sweep.sh queues and two launched separately. Every one is dumped
# on the 64-epoch stride and ALL of them are kept: the analysis iterates a run's
# whole directory to build its "ever crossed Mars" mask, so an epoch that no
# call site names is still read.
#
# Checked afterwards by running analysis.ipynb and diffing its output against
# /tmp/nb_before.out, which was taken while the previous copy of this data was
# intact.
set -u
cd /home/jumqyc/Works/作业/SelfStudy/Kirkwood
B=build-o3/kirkwood_gpu
mkdir -p data/sweep

# name  years  e_max  saturn  eccentric  binary
run_one() {
    local name=$1 years=$2 emax=$3 sat=$4 ecc=$5 bin=$6
    local steps
    steps=$(awk -v y="$years" 'BEGIN{printf "%d", y * 365.25 / 10}')
    local every=$((steps / 64))
    echo "=== $name: $years yr, e_max=$emax, saturn=$sat, eccentric=$ecc ==="
    if [ "$name" = "ecc_dt2" ]; then
        steps=$(awk -v y="$years" 'BEGIN{printf "%d", y * 365.25 / 2}')
        every=$((steps / 64))
    fi
    "$bin" 100000 "$steps" "data/sweep/$name" "$every" "$emax" "$sat" 114514 "$ecc" \
        > "data/sweep/$name.out" 2>&1
    echo "  $(ls "data/sweep/$name"/epoch_*.bin | wc -l) epochs"
}

run_one e010_long  3000000 0.10 0 0 $B
run_one e005       1000000 0.05 0 0 $B
run_one e020       1000000 0.20 0 0 $B
run_one e030       1000000 0.30 0 0 $B
run_one e010sat    1000000 0.10 1 0 $B
run_one ecc         300000 0.10 1 1 $B
run_one ecc_dt2      20000 0.10 1 1 build-o3/kirkwood_gpu_dt2

echo "ALL DONE"
