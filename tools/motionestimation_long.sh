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
# initialize "walking" 1 1000 # debug
INITTRAININGITERS=10000
train() {
    ok=true
    for i in $(seq $(expr $2 + 1) $(expr $2 + $6 - 1)); do
        EXISTSPATH="output/$1/$4/frame$i/point_cloud/iteration_$3/point_cloud.ply"
        if [ -e "$EXISTSPATH" ]; then
            echo "(skip) exists: $EXISTSPATH"
            ok=true
        else
            echo "not exists: $EXISTSPATH"
            ok=false
            break
        fi
    done
    if [ "$ok" = true ]; then
        echo "(skip) all exists: output/$1/frame<$2-$(expr $2 + $6 - 1)>/point_cloud/iteration_$3/point_cloud.ply"
        return
    fi
    # echo \
    python -m instantsplatstream.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $3 \
        --pipeline $4 $5 \
        -n $6 -b $7 \
        --load_camera $8
}
train "walking" 1 1000 track/propagate-dot-cotracker3 "" 100 8 "output/walking/frame1/cameras.json" # debug
