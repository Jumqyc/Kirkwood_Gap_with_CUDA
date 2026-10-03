#!/bin/bash
# Redraws every frame of the video at 4K from the per-frame .npz files.
#
# Four shards per segment, run as separate processes rather than through a
# Python process pool: rendering is CPU-bound and matplotlib does not release
# the GIL, so threads would not help, and in this environment a pool cannot even
# start because creating its semaphore is denied.
#
# The segments keep their own output directories because their step numbers
# overlap -- s1 is dt = 5 days and s2 is dt = 20 -- so a single directory would
# have frames overwriting each other.
set -u
cd /home/jumqyc/Works/作业/SelfStudy/Kirkwood
PY=.venv/bin/python
SHARDS=4

rm -rf data/video/frames4k
for s in s1 s2 s3 s4; do
    out=data/video/frames4k/$s
    mkdir -p "$out"
    cp data/video/work/baseline.npy "$out/"
    echo "=== $s ==="
    for i in $(seq 0 $((SHARDS - 1))); do
        $PY scripts/render_from_npz.py data/video/work/$s/frames "$out" \
            --dpi 200 --vmax 16000 --shard "$i/$SHARDS" \
            > "data/video/frames4k/$s.shard$i.log" 2>&1 &
    done
    wait
    echo "  $s: $(ls "$out"/*.png 2>/dev/null | wc -l) 帧"
done
echo "ALL DONE"
