#!/bin/bash

extract_SIGA2025VVC() {
    ROOT=data/SIGA2025VVC-Dataset/$1
    rm -rf $ROOT/images
    tar -zxvf $ROOT/images.tar.gz -C $ROOT
    rm -rf $ROOT/masks
    tar -zxvf $ROOT/masks.tar.gz -C $ROOT
    for frame in $(seq 1 $2); do
        rm -rf $ROOT/frame$frame
    done
    python tools/parse_camera_SIGA2025VVC.py --path $ROOT --n_frames $2
}
extract_SIGA2025VVC compression/test/004_1_seq1 300
extract_SIGA2025VVC compression/test/006_1_seq1 300
extract_SIGA2025VVC compression/test/007_0_seq1 300
extract_SIGA2025VVC compression/test/008_2_seq1 300
extract_SIGA2025VVC compression/test/008_2_seq1 300
extract_SIGA2025VVC compression/test/011_0_seq1 300
extract_SIGA2025VVC compression/val/001_1_seq0 300
extract_SIGA2025VVC compression/val/012_0_seq0 300

ITERS=10000

# hyperparams
ARGSCOMMON=$ARGSCOMMON" --with_scale_reg"
ARGSCOMMON=$ARGSCOMMON" --empty_cache_every_step"
ARGSCOMMON=$ARGSCOMMON" -oscale_reg_thr_scale=0.2"
ARGSCOMMON=$ARGSCOMMON" -odensify_percent_too_big=0.15"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_init=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_final=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_from_iter=4000"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_kernel_radius=32"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_stride=16"
ARGSCOMMON=$ARGSCOMMON" -odepth_resize=577"
ARGSCOMMON=$ARGSCOMMON" -omercy_type='redundancy_opacity_opacity'"
ARGSCOMMON=$ARGSCOMMON" -oimportance_score_resize=1280"

# steps
ARGSSTEPS=$ARGSSTEPS" --save_iterations=10000"
ARGSSTEPS=$ARGSSTEPS" -oposition_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocull_at_steps=[9000]"
ARGSSTEPS=$ARGSSTEPS" -oscale_reg_from_iter=500"
ARGSSTEPS=$ARGSSTEPS" -odepth_l1_weight_max_steps=10000"

# steps for camera
ARGSCAMERA=$ARGSCAMERA" -ocamera_position_lr_max_steps=10000"
ARGSCAMERA=$ARGSCAMERA" -ocamera_rotation_lr_max_steps=10000"
ARGSCAMERA=$ARGSCAMERA" -ocamera_exposure_lr_max_steps=10000"

# steps for densify
ARGSDENSIFY=$ARGSDENSIFY" -odensify_from_iter=1000"
ARGSDENSIFY=$ARGSDENSIFY" -odensify_until_iter=7500"
ARGSDENSIFY=$ARGSDENSIFY" -odensify_interval=100"
ARGSDENSIFY=$ARGSDENSIFY" -oprune_from_iter=2000"
ARGSDENSIFY=$ARGSDENSIFY" -oprune_until_iter=7000"
ARGSDENSIFY=$ARGSDENSIFY" -oprune_interval=1000"
ARGSDENSIFY=$ARGSDENSIFY" -oopacity_reset_from_iter=3000"
ARGSDENSIFY=$ARGSDENSIFY" -oopacity_reset_until_iter=5000"
ARGSDENSIFY=$ARGSDENSIFY" -oopacity_reset_interval=1000"
ARGSDENSIFY=$ARGSDENSIFY" -oimportance_prune_from_iter=2000"
ARGSDENSIFY=$ARGSDENSIFY" -oimportance_prune_until_iter=8500"
ARGSDENSIFY=$ARGSDENSIFY" -oimportance_prune_interval=100"

train_camera() {
    MODE=camera-densify-prune-shculling
    ARGS="$ARGSCOMMON $ARGSSTEPS $ARGSDENSIFY $ARGSCAMERA"
    EXISTSPATH="output/$1/frame$2/point_cloud/iteration_$ITERS/point_cloud.ply"
    if [ -e "$EXISTSPATH" ]; then
        echo "(skip) exists: $EXISTSPATH"
        return
    fi
    echo "not exists: $EXISTSPATH"
    # echo \
    python -m reduced_3dgs.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2/camera \
        --mode $MODE \
        -i $ITERS $ARGS
}

train_scene() {
    MODE=densify-prune-shculling
    ARGS="$ARGSCOMMON $ARGSSTEPS $ARGSDENSIFY"
    EXISTSPATH="output/$1/frame$2/point_cloud/iteration_$ITERS/point_cloud.ply"
    if [ -e "$EXISTSPATH" ]; then
        echo "(skip) exists: $EXISTSPATH"
        return
    fi
    echo "not exists: $EXISTSPATH"
    # echo \
    python -m reduced_3dgs.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2/camera \
        --mode $MODE \
        -i $ITERS $ARGS \
        --load_camera "output/$1/frame$2/camera/cameras.json"
}

train() {
    train_camera $1 $2
    train_scene $1 $2
}

train SIGA2025VVC-Dataset/compression/test/004_1_seq1 1
train SIGA2025VVC-Dataset/compression/test/006_1_seq1 1
train SIGA2025VVC-Dataset/compression/test/007_0_seq1 1
train SIGA2025VVC-Dataset/compression/test/008_2_seq1 1
train SIGA2025VVC-Dataset/compression/test/008_2_seq1 1
train SIGA2025VVC-Dataset/compression/test/011_0_seq1 1
train SIGA2025VVC-Dataset/compression/val/001_1_seq0 1
train SIGA2025VVC-Dataset/compression/val/012_0_seq0 1
