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
        echo "(skip) all exists: output/$1/frame<$2-$(expr $2 + $7 - 1)>/point_cloud/iteration_$3/point_cloud.ply"
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
# train "walking" 1 1000 track/propagate-dot-cotracker3 "" 8 100 "output/walking/frame1/cameras.json" # debug
initialize_and_train_video_allmethods() {
    CAMERAS="output/$1/frame$2/cameras.json"
    train $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5 $6 "$CAMERAS"
    train $1 $2 $3 refine/base-base-dot-cotracker3 "-o rescale_factor=$4" $5 $6 "$CAMERAS"
    train $1 $2 $3 train/regularized "-o neighbors=20" $5 $6 "$CAMERAS"
    train $1 $2 $3 train/base "" $5 $6 "$CAMERAS"
    train $1 $2 $3 train/hexplane "" $5 $6 "$CAMERAS"
    train $1 $2 $3 train/regularizedhexplane "-o neighbors=20" $5 $6 "$CAMERAS"
}
initialize_and_train_video_allmethods walking 1 1000 0.3 8 75 # debug
initialize_and_train_video_allmethods taekwondo 1 1000 0.3 8 101
initialize_and_train_video_allmethods boxing 1 1000 0.3 8 71

initialize_and_train_video_allmethods coffee_martini 1 1000 0.3 8 300
initialize_and_train_video_allmethods cook_spinach 1 1000 0.3 8 300
initialize_and_train_video_allmethods cut_roasted_beef 1 1000 0.3 8 300
initialize_and_train_video_allmethods flame_salmon_1 1 1000 0.3 8 1200
initialize_and_train_video_allmethods flame_steak 1 1000 0.3 8 300
initialize_and_train_video_allmethods sear_steak 1 1000 0.3 8 300

initialize_and_train_video_allmethods discussion 1 1000 0.5 8 300
initialize_and_train_video_allmethods stepin 1 1000 0.5 8 300
initialize_and_train_video_allmethods trimming 1 1000 0.5 8 300
initialize_and_train_video_allmethods vrheadset 1 1000 0.5 8 300

initialize_and_train_video_allmethods basketball 1 1000 1.0 8 150
initialize_and_train_video_allmethods boxes 1 1000 1.0 8 150
initialize_and_train_video_allmethods football 1 1000 1.0 8 150
initialize_and_train_video_allmethods juggle 1 1000 1.0 8 150
initialize_and_train_video_allmethods softball 1 1000 1.0 8 150
initialize_and_train_video_allmethods tennis 1 1000 1.0 8 150
