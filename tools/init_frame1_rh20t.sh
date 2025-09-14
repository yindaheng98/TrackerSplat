#!/bin/bash
ITERS=10000
MODE=camera-shculling
ARGS=""
train() {
    EXISTSPATH="output/$1/frame$2/point_cloud/iteration_$ITERS/point_cloud.ply"
    if [ -e "$EXISTSPATH" ]; then
        echo "(skip) exists: $EXISTSPATH"
        return
    fi
    echo "not exists: $EXISTSPATH"
    # echo \
    python -m reduced_3dgs.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2 \
        --mode $MODE \
        -i $ITERS $ARGS
}
# train walking 1 # debug

ARGSCOMMON=""
# hyperparams
ARGSCOMMON=$ARGSCOMMON" --with_scale_reg"
ARGSCOMMON=$ARGSCOMMON" --empty_cache_every_step"
ARGSCOMMON=$ARGSCOMMON" -oscale_reg_thr_scale=0.2"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_init=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_final=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_from_iter=4000"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_kernel_radius=32"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_stride=16"
ARGSCOMMON=$ARGSCOMMON" -odepth_resize=577"

ARGSSTEPS=""
# steps
ARGSSTEPS=$ARGSSTEPS" --save_iterations=10000"
ARGSSTEPS=$ARGSSTEPS" -oposition_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocull_at_steps=[9000]"
ARGSSTEPS=$ARGSSTEPS" -oscale_reg_from_iter=500"
ARGSSTEPS=$ARGSSTEPS" -odepth_l1_weight_max_steps=10000"
# steps for camera
ARGSSTEPS=$ARGSSTEPS" -ocamera_position_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocamera_rotation_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocamera_exposure_lr_max_steps=10000"

ARGS="$ARGSCOMMON $ARGSSTEPS $ARGSDENSIFY"

train RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003 1
