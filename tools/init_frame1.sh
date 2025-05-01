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

ARGS="$ARGSCOMMON $ARGSSTEPS $ARGSDENSIFY"

train coffee_martini 1
train cook_spinach 1
train cut_roasted_beef 1
train flame_salmon_1 1
train flame_steak 1
train sear_steak 1

train discussion 1
train stepin 1
train trimming 1
train vrheadset 1

train boxing 1
train taekwondo 1
train walking 1

# ARGS="$ARGS --no_depth_data"
train basketball 1
train boxes 1
train football 1
train juggle 1
train softball 1
train tennis 1
