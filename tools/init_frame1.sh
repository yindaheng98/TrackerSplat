#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)
INITARGS=""
INITMODE=colmap-dense
initialize() {
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame$2 \
        --initializer $INITMODE \
        -o "colmap_executable='$COLMAP_EXECUTABLE'" $INITARGS
}
# initialize walking 1 # debug

INITTRAININGITERS=10000
INITTRAININGMODE=camera-densify-prune-shculling
INITTRAININGARGS=""
train() {
    # echo \
    python -m reduced_3dgs.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2 \
        --mode $INITTRAININGMODE \
        -i $INITTRAININGITERS $INITTRAININGARGS
}
# train walking 1 # debug

init_and_train() {
    EXISTSPATH="output/$1/frame$2/point_cloud/iteration_$INITTRAININGITERS/point_cloud.ply"
    if [ -e "$EXISTSPATH" ]; then
        echo "(skip) exists: $EXISTSPATH"
        return
    fi
    echo "not exists: $EXISTSPATH"
    initialize $1 $2
    train $1 $2
}

INITTRAININGARGSCOMMON=""
INITTRAININGARGSCOMMON=$INITTRAININGARGSCOMMON" --empty_cache_every_step"
INITTRAININGARGSCOMMON=$INITTRAININGARGSCOMMON" --with_scale_reg"
INITTRAININGARGSCOMMON=$INITTRAININGARGSCOMMON" -oscale_reg_thr_scale=0.5"
INITTRAININGARGSCOMMON=$INITTRAININGARGSCOMMON" -odepth_l1_weight_init=1.0"
INITTRAININGARGSCOMMON=$INITTRAININGARGSCOMMON" -odepth_l1_weight_final=1.0"

INITTRAININGARGSDENSIFY=""
INITTRAININGARGSDENSIFY=$INITTRAININGARGSDENSIFY" -odensify_percent_too_big=0.5"

INITTRAININGARGSSTEPS=""
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" --save_iterations=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oposition_lr_max_steps=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -ocamera_position_lr_max_steps=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -ocamera_rotation_lr_max_steps=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -ocamera_exposure_lr_max_steps=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -odensify_from_iter=1000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -odensify_until_iter=8000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -odensify_interval=100"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oprune_from_iter=2000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oprune_until_iter=8000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oprune_interval=500"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oopacity_reset_from_iter=4000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oopacity_reset_until_iter=8000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oopacity_reset_interval=500"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -ocull_at_steps=[9000]"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -odepth_l1_weight_max_steps=10000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -odepth_from_iter=1000"
INITTRAININGARGSSTEPS=$INITTRAININGARGSSTEPS" -oscale_reg_from_iter=5000"

INITTRAININGARGS="$INITTRAININGARGSCOMMON $INITTRAININGARGSDENSIFY $INITTRAININGARGSSTEPS"
init_and_train walking 1
init_and_train taekwondo 1
init_and_train boxing 1

INITARGS="-o use_fused=True"
init_and_train basketball 1
init_and_train boxes 1
init_and_train football 1
init_and_train juggle 1
init_and_train softball 1
init_and_train tennis 1
INITARGS=""
INITTRAININGARGS=""
