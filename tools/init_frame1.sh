#!/bin/bash
ITERS=10000
MODE=camera-densify-prune-shculling
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
ARGSCOMMON=$ARGSCOMMON" --empty_cache_every_step"
ARGSCOMMON=$ARGSCOMMON" --with_scale_reg"
ARGSCOMMON=$ARGSCOMMON" -oscale_reg_thr_scale=0.5"
ARGSCOMMON=$ARGSCOMMON" -odepth_from_iter=500"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_init=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_l1_weight_final=1.0"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_kernel_radius=32"
ARGSCOMMON=$ARGSCOMMON" -odepth_local_relative_stride=16"
ARGSCOMMON=$ARGSCOMMON" -odepth_resize=577"

ARGSDENSIFY=""
ARGSDENSIFY=$ARGSDENSIFY" -odensify_percent_too_big=0.3"

ARGSSTEPS=""
ARGSSTEPS=$ARGSSTEPS" --save_iterations=10000"
ARGSSTEPS=$ARGSSTEPS" -oposition_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocamera_position_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocamera_rotation_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -ocamera_exposure_lr_max_steps=10000"
ARGSSTEPS=$ARGSSTEPS" -odensify_from_iter=1000"
ARGSSTEPS=$ARGSSTEPS" -odensify_until_iter=8000"
ARGSSTEPS=$ARGSSTEPS" -odensify_interval=100"
ARGSSTEPS=$ARGSSTEPS" -oprune_from_iter=2000"
ARGSSTEPS=$ARGSSTEPS" -oprune_until_iter=8000"
ARGSSTEPS=$ARGSSTEPS" -oprune_interval=500"
ARGSSTEPS=$ARGSSTEPS" -oopacity_reset_from_iter=4000"
ARGSSTEPS=$ARGSSTEPS" -oopacity_reset_until_iter=8000"
ARGSSTEPS=$ARGSSTEPS" -oopacity_reset_interval=500"
ARGSSTEPS=$ARGSSTEPS" -ocull_at_steps=[9000]"
ARGSSTEPS=$ARGSSTEPS" -oscale_reg_from_iter=500"
ARGSSTEPS=$ARGSSTEPS" -odepth_l1_weight_max_steps=10000"

ARGS="$ARGSCOMMON $ARGSDENSIFY $ARGSSTEPS"
train walking 1
train taekwondo 1
train boxing 1

train basketball 1
train boxes 1
train football 1
train juggle 1
train softball 1
train tennis 1
