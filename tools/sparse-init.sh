#!/bin/bash
COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
# COLMAP_EXECUTABLE=$(which colmap)
initialize() {
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-sparse \
        -o "colmap_executable='$COLMAP_EXECUTABLE'"
    for i in $(seq 2 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer colmap-sparse \
            -o "load_camera='./data/$1/frame1'" \
            -o "colmap_executable='$COLMAP_EXECUTABLE'"
    done
}

# Meeting room datasets
# initialize "stepin" 300
initialize "walking" 75

initialize_dynamic3dgs() {
    # echo \
    python tools/dynamic3dgs.py \
        --path data/$1 \
        --colmap_executable $COLMAP_EXECUTABLE \
        --n_frames $2
}

initialize_dynamic3dgs basketball 150
initialize_dynamic3dgs boxes 150
initialize_dynamic3dgs football 150
initialize_dynamic3dgs juggle 150
initialize_dynamic3dgs softball 150
initialize_dynamic3dgs tennis 150
