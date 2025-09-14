#!/bin/bash

initialize() {
    rm -rf data/$1/frame*
    python tools/extract_rh20t.py --path data/$1
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer dust3r
}

initialize RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003
