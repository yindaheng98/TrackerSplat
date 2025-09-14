#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)

# Dataset without pose
initialize_nopose() {
    eval before_initialize_nopose_$MODE $1 $2 # remove the old data
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-sparse \
        -o "colmap_executable='$COLMAP_EXECUTABLE'"
    # n=0
    for i in $(seq 2 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer nodepth-colmap-sparse \
            -o "load_camera='./data/$1/frame1'" \
            -o "colmap_executable='$COLMAP_EXECUTABLE'" \
            --device cuda
        #     --device cpu &
        # n=$(expr $n + 1)
        # if [ $n -eq 16 ]; then
        #     wait
        #     n=0
        # fi
    done
    # wait
    echo Done $MODE $1 $2
}

before_initialize_nopose_rh20t() {
    rm -rf data/$1/frame*
    python tools/extract_rh20t.py --path data/$1
}
MODE=rh20t
initialize_nopose RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003
