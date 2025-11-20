#!/bin/bash
INITTRAININGITERS=10000
REFININGITERS=1000
train() {
    ok=true
    skip=0
    for i in $(seq $(expr $2 + 1) $(expr $2 + $6 - 1)); do
        EXISTSPATH="output/$1/$3/frame$i/point_cloud/iteration_$REFININGITERS/point_cloud.ply"
        if [ -e "$EXISTSPATH" ]; then
            echo "(skip) exists: $EXISTSPATH"
            ok=true
            if [ $(( (i-1) % ("$5" - 1) )) -eq 0 ]; then
              skip=$((skip + "$5" - 1))
            fi
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
    echo "(skip) exists: output/$1/$3/frame<$2-$(expr $2 + $skip)>/point_cloud/iteration_$REFININGITERS/point_cloud.ply"
    # echo \
    python -m trackersplat.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $REFININGITERS \
        --pipeline $3 $4 \
        -b $5 -n $6 --skip $skip \
        --load_camera $7 \
        --with_image_mask
}

REFINEARGS="$REFINEARGS -omask_input=False"
REFINEARGS="$REFINEARGS -omask_output=False"

REFINEARGS="$REFINEARGS -rposition_lr_init=0.0016"
REFINEARGS="$REFINEARGS -rposition_lr_max_steps=$REFININGITERS"
REFINEARGS="$REFINEARGS -rfeature_lr=0.0" # stable SH
REFINEARGS="$REFINEARGS -rscaling_lr=0.00005" # stable scaling
REFINEARGS="$REFINEARGS -rmask_mode='bg_color'"
REFINEARGS="$REFINEARGS -rbg_color='random'"

TRAINARGS="$TRAINARGS -oposition_lr_init=0.0016"
TRAINARGS="$TRAINARGS -oposition_lr_max_steps=$REFININGITERS"
TRAINARGS="$TRAINARGS -ofeature_lr=0.0" # stable SH
TRAINARGS="$TRAINARGS -omask_mode='bg_color'"
TRAINARGS="$TRAINARGS -obg_color='random'"

PATCHARGS="$PATCHARGS --patcher=densify"
PATCHARGS="$PATCHARGS -ppatch_every_n_frames=1"
PATCHARGS="$PATCHARGS -ppatch_every_n_updates=1"
PATCHARGS="$PATCHARGS -piteration=$REFININGITERS"
PATCHARGS="$PATCHARGS -ptrainer='patch-densify'"
TRAINARGS="$TRAINARGS -pposition_lr_init=0.0016"
TRAINARGS="$TRAINARGS -pposition_lr_max_steps=$REFININGITERS"
PATCHARGS="$PATCHARGS -pfeature_lr=0.0" # stable SH
PATCHARGS="$PATCHARGS -pscaling_lr=0.00005" # stable scaling
PATCHARGS="$PATCHARGS -pmask_mode='bg_color'"
PATCHARGS="$PATCHARGS -pbg_color='random'"

initialize_and_train_video_allmethods() {
    CAMERAS="output/$1/frame$2/cameras.json"
    # train $1 $2 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$3 $REFINEARGS $PATCHARGS" 5 $4 "$CAMERAS"
    train $1 $2 train/regularized "-o neighbors=8 -o loss_weight_overall=0.1 $TRAINARGS $PATCHARGS" 2 $4 "$CAMERAS"
}

initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/004_1_seq1 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/006_1_seq1 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/007_0_seq1 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/008_2_seq1 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/011_0_seq1 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/001_1_seq0 1 0.25 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/012_0_seq0 1 0.25 300
