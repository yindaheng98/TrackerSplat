#!/bin/bash

initialize() {
    rm -rf data/$1/frame*
    python tools/extract_rh20t.py --path data/$1
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer dust3r
    n=2
    while [ -d "data/$1/frame$n" ]; do
        rm -rf data/$1/frame$n/sparse
        cp -r data/$1/frame1/sparse data/$1/frame$n
        n=$(expr $n + 1)
    done
}
# initialize RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003 # debug

for s in data/RH20T_cfg3/task_*_user_*_scene_*_cfg_0003; do
    initialize ${s:5}
done