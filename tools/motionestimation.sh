#!/bin/bash
COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
# COLMAP_EXECUTABLE=$(which colmap)
INITARGS=""
initialize() {
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
# initialize "walking" 10 1000 # debug
train() {
    # echo \
    python -m instantsplatstream.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $3 -i $3 \
        --pipeline $4 --tracking_rescale $5 \
        -n $(expr $6 - 1) -b $6
}
# train "walking" 10 1000 track/propagate-dot-cotracker3 0.3 10 # debug
initialize_and_train_clip() {
    initialize $1 $2 $3
    train $1 $2 $3 $4 $5 $6
}
# initialize_and_train_clip "walking" 10 1000 track/propagate-dot-cotracker3 0.3 10 # debug
initialize_and_train_clip_allmethods() {
    initialize $1 $2 $3
    train $1 $2 $3 refine/regularized-propagate-dot-cotracker3 $4 $5
    train $1 $2 $3 refine/base-propagate-dot-cotracker3 $4 $5
    train $1 $2 $3 train/regularized $4 $5
    train $1 $2 $3 train/base $4 $5
}
# initialize_and_train_clip_allmethods "walking" 10 1000 0.3 10 # debug
initialize_and_train_allvideo_allmethods() {
    for i in $(seq 0 $2); do
        initialize_and_train_clip_allmethods $1 $(expr $i \* $5 + 1) $3 $4 $5
    done
}
# initialize_and_train_allvideo_allmethods "walking" 7 1000 0.3 10 # debug
INITARGS="-o use_fused=True"
initialize_and_train_allvideo_allmethods "basketball" 15 1000 1.0 10 # debug
