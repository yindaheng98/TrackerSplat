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
        echo "(skip) all exists: output/$1/$4/frame<$2-$(expr $2 + $7 - 1)>/point_cloud/iteration_$3/point_cloud.ply"
        return
    fi
    # echo \
    python -m trackersplat.motionestimation \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $3 -rposition_lr_max_steps=$3 \
        --pipeline $4 $5 \
        -b $6 -n $7 $REFININGARGS \
        --load_camera $8 \
        --with_image_mask -omask_input=False -omask_output=False
}
# train "walking" 1 1000 track/propagate-dot-cotracker3 "" 9 100 "output/walking/frame1/cameras.json" # debug
initialize_and_train_video_allmethods() {
    CAMERAS="output/$1/frame$2/cameras.json"
    train $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5 $6 "$CAMERAS"
}
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/004_1_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/006_1_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/007_0_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/008_2_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/008_2_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/test/011_0_seq1 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/001_1_seq0 1 1000 0.2 9 300
initialize_and_train_video_allmethods SIGA2025VVC-Dataset/compression/val/012_0_seq0 1 1000 0.2 9 300
