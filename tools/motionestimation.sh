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
# initialize "walking" 10 1000 # debug
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
        -n $(expr $6 - 1) -b $6 \
        --load_camera $7
}
# train "walking" 10 1000 track/propagate-dot-cotracker3 0.3 10 # debug
initialize_and_train_clip_allmethods() {
    initialize $1 $2 $INITTRAININGITERS
    CAMERAS="output/$1/frame$2/cameras.json"
    train $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5 "$CAMERAS"
    train $1 $2 $3 refine/base-base-dot-cotracker3 "-o rescale_factor=$4" $5 "$CAMERAS"
    train $1 $2 $3 train/regularized "-o neighbors=20" $5 "$CAMERAS"
    train $1 $2 $3 train/base "" $5 "$CAMERAS"
    train $1 $2 $3 train/regularizedhexplane "-o neighbors=20" $5 "$CAMERAS"
    train $1 $2 $3 train1step/regularized "-o neighbors=20" $5 "$CAMERAS"
    train $1 $2 $3 train1step/base "" $5 "$CAMERAS"
    train $1 $2 $3 train1step/regularizedhexplane "-o neighbors=20" $5 "$CAMERAS"
}
# initialize_and_train_clip_allmethods "walking" 10 1000 0.3 10 # debug
initialize_and_train_allvideo_allmethods() {
    for i in $(seq 0 $2); do
        initialize_and_train_clip_allmethods $1 $(expr $i \* $5 + 1) $3 $4 $5
    done
}
initialize_and_train_allvideo_allmethods walking 6 1000 0.3 10
initialize_and_train_allvideo_allmethods taekwondo 10 1000 0.3 10

initialize_and_train_allvideo_allmethods coffee_martini 30 1000 0.3 10
initialize_and_train_allvideo_allmethods cook_spinach 30 1000 0.3 10
initialize_and_train_allvideo_allmethods cut_roasted_beef 30 1000 0.3 10
initialize_and_train_allvideo_allmethods flame_salmon_1 120 1000 0.3 10
initialize_and_train_allvideo_allmethods flame_steak 30 1000 0.3 10
initialize_and_train_allvideo_allmethods sear_steak 30 1000 0.3 10

INITARGS="-o use_fused=True"
initialize_and_train_allvideo_allmethods discussion 30 1000 0.5 10
initialize_and_train_allvideo_allmethods stepin 30 1000 0.5 10
initialize_and_train_allvideo_allmethods trimming 30 1000 0.5 10
initialize_and_train_allvideo_allmethods vrheadset 30 1000 0.5 10

INITARGS="-o use_fused=True"
initialize_and_train_allvideo_allmethods basketball 15 1000 1.0 10
initialize_and_train_allvideo_allmethods boxes 15 1000 1.0 10
initialize_and_train_allvideo_allmethods football 15 1000 1.0 10
initialize_and_train_allvideo_allmethods juggle 15 1000 1.0 10
initialize_and_train_allvideo_allmethods softball 15 1000 1.0 10
initialize_and_train_allvideo_allmethods tennis 15 1000 1.0 10
