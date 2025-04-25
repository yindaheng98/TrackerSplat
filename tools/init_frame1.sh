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
train() {
    # echo \
    python -m reduced_3dgs.train \
        -s data/$1/frame$2 \
        -d output/$1/frame$2 \
        --mode $INITTRAININGMODE \
        -i $INITTRAININGITERS $INITTRAININGARGS
}
train walking 1 # debug
