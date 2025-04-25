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
INITTRAININGARGS=$INITTRAININGARGS" --empty_cache_every_step"
INITTRAININGARGS=$INITTRAININGARGS" --save_iterations=10000"
INITTRAININGARGS=$INITTRAININGARGS" -oposition_lr_max_steps=10000"
INITTRAININGARGS=$INITTRAININGARGS" -ocamera_position_lr_max_steps=10000"
INITTRAININGARGS=$INITTRAININGARGS" -ocamera_rotation_lr_max_steps=10000"
INITTRAININGARGS=$INITTRAININGARGS" -ocamera_exposure_lr_max_steps=10000"
INITTRAININGARGS=$INITTRAININGARGS" -odensify_from_iter=2000"
INITTRAININGARGS=$INITTRAININGARGS" -odensify_until_iter=8000"
INITTRAININGARGS=$INITTRAININGARGS" -odensify_interval=100"
INITTRAININGARGS=$INITTRAININGARGS" -odensify_percent_too_big=0.5"
INITTRAININGARGS=$INITTRAININGARGS" -oprune_from_iter=2000"
INITTRAININGARGS=$INITTRAININGARGS" -oprune_until_iter=8000"
INITTRAININGARGS=$INITTRAININGARGS" -oprune_interval=100"
INITTRAININGARGS=$INITTRAININGARGS" -oprune_percent_too_big=1.0"
INITTRAININGARGS=$INITTRAININGARGS" -oopacity_reset_from_iter=4000"
INITTRAININGARGS=$INITTRAININGARGS" -oopacity_reset_until_iter=8000"
INITTRAININGARGS=$INITTRAININGARGS" -oopacity_reset_interval=1000"
INITTRAININGARGS=$INITTRAININGARGS" -ocull_at_steps=[9000]"
INITTRAININGARGS=$INITTRAININGARGS" --with_scale_reg"
INITTRAININGARGS=$INITTRAININGARGS" -oscale_reg_thr_scale=0.5"
INITTRAININGARGS=$INITTRAININGARGS" -oscale_reg_from_iter=5000"
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