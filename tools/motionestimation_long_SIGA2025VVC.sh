#!/bin/bash
INITTRAININGITERS=10000
REFININGITERS=1000
train() {
    ok=true
    for i in $(seq $(expr $2 + 1) $(expr $2 + $6 - 1)); do
        EXISTSPATH="output/$1/$3/frame$i/point_cloud/iteration_$REFININGITERS/point_cloud.ply"
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
        echo "(skip) all exists: output/$1/$3/frame<$2-$(expr $2 + $6 - 1)>/point_cloud/iteration_$REFININGITERS/point_cloud.ply"
        return
    fi
    # echo \
    python -m trackersplat.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $REFININGITERS \
        --pipeline $3 $4 \
        -b $5 -n $6 \
        --load_camera $7 \
        --with_image_mask
}

REFINEARGS="$REFINEARGS -omask_input=False"
REFINEARGS="$REFINEARGS -omask_output=False"

REFINEARGS="$REFINEARGS -rposition_lr_init=0.0016"
REFINEARGS="$REFINEARGS -rposition_lr_max_steps=$REFININGITERS"
REFINEARGS="$REFINEARGS -rscaling_lr=0.000001"
REFINEARGS="$REFINEARGS -ropacity_lr=0.000001"
REFINEARGS="$REFINEARGS -rmask_mode='bg_color'"
REFINEARGS="$REFINEARGS -rbg_color='random'"

TRAINARGS="$TRAINARGS -oposition_lr_init=0.0016"
TRAINARGS="$TRAINARGS -oposition_lr_max_steps=$REFININGITERS"
TRAINARGS="$TRAINARGS -oscaling_lr=0.000001"
TRAINARGS="$TRAINARGS -oopacity_lr=0.000001"
TRAINARGS="$TRAINARGS -omask_mode='bg_color'"
TRAINARGS="$TRAINARGS -orbg_color='random'"

initialize_and_train_video_allmethods() {
    CAMERAS="output/$1/frame$2/cameras.json"
    train $1 $2 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4 $REFINEARGS" $4 $5 "$CAMERAS"
    train $1 $2 train/regularized "-o neighbors=8 -o loss_weight_overall=0.1 $TRAINARGS" $4 $5 "$CAMERAS"
}
# BATCHSIZE=9
BATCHSIZE=2
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/004_1_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/006_1_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/007_0_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/008_2_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/008_2_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/011_0_seq1 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/001_1_seq0 1 0.25 $BATCHSIZE 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/012_0_seq0 1 0.25 $BATCHSIZE 300
