#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)
INITARGS=""
initialize() {
    EXISTSPATH="output/$1/frame$2/point_cloud/iteration_$3/point_cloud.ply"
    if [ -e "$EXISTSPATH" ]; then
        echo "(skip) exists: $EXISTSPATH"
        return
    fi
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame$2 \
        --initializer colmap-dense \
        -o "colmap_executable='$COLMAP_EXECUTABLE'" $INITARGS
    # echo \
    python -m instantsplat.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2 \
        -i $3
}
initialize "walking" 1 1000 # debug
