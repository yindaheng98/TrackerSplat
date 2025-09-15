#!/bin/bash
# rm -rf data/RH20T_cfg5 && tar -zxvf data/RH20T_cfg5.tar.gz -C data
initialize() {
    rm -rf data/$1/frame*
    python tools/extract_rh20t.py --path data/$1 --except 104422070042 --except 135122079702 --except 104422071090
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
# initialize RH20T_cfg5/task_0001_user_0016_scene_0002_cfg_0005 # debug

for s in data/RH20T_cfg5/task_*_user_*_scene_*_cfg_0005; do
    initialize ${s:5}
done