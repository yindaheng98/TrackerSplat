#!/bin/bash
INITTRAININGITERS=10000
# initialize "walking" 1 # debug
REFININGARGS=""
REFININGARGS=$REFININGARGS" -rscaling_lr=0.000001"
train() {
    ok=true
    for i in $(seq $(expr $2 + 1) $(expr $2 + $7 - 1)); do
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
        echo "(skip) all exists: output/$1/$4/frame<$2-$(expr $2 + $7 - 1)>/point_cloud/iteration_$3/point_cloud.ply"
        return
    fi
    # echo \
    python -m instantsplatstream.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $3 -rposition_lr_max_steps=$3 \
        --pipeline $4 $5 \
        -b $6 -n $7 $REFININGARGS \
        --load_camera $8
}
# train "walking" 1 1000 track/propagate-dot-cotracker3 "" 9 100 "output/walking/frame1/cameras.json" # debug
n_frames() {
    n=0
    while [ -d "data/$1/frame$(expr $n + 1)" ]; do
        n=$(expr $n + 1)
    done
    echo $n
}
initialize_and_train_video_allmethods() {
    CAMERAS="output/$1/frame$2/cameras.json"
    N=$(n_frames $1)
    train $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5 $N "$CAMERAS"
    train $1 $2 $3 refine/base-base-dot-cotracker3 "-o rescale_factor=$4" $5 $N "$CAMERAS"
    train $1 $2 $3 train/regularized "-o neighbors=20" $5 $N "$CAMERAS"
    train $1 $2 $3 train/hicom "" $5 $N "$CAMERAS"
    train $1 $2 $3 train/hexplane "" $5 $N "$CAMERAS"
    train $1 $2 $3 train/regularizedhexplane "-o neighbors=20" $5 $N "$CAMERAS"
}
initialize_and_train_video_allmethods RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003 1 1000 1.0 9 # debug
