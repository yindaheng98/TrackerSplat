#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)

initialize() {
    eval before_initialize_$MODE $1 $2 # remove the old data
    # echo \
    python tools/parse_camera.py \
        --mode $MODE \
        --path data/$1 \
        --colmap_executable $COLMAP_EXECUTABLE \
        --n_frames $2 # parse the given camera parameters
    # n=0
    for i in $(seq 1 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer colmap-sparse \
            -o "colmap_executable='$COLMAP_EXECUTABLE'" \
            --device cuda # sparse initialization
        #     --device cpu &
        # n=$(expr $n + 1)
        # if [ $n -eq 8 ]; then
        #     wait
        #     n=0
        # fi
    done
    # wait
}

before_initialize_n3dv() {
    for i in $(seq 1 $2); do
        mv data/$1/frame$i data/$1/tmpframe$i
        mkdir -p data/$1/frame$i
        mv data/$1/tmpframe$i/input data/$1/frame$i/input
        rm -rf data/$1/tmpframe$i
    done
}
MODE=n3dv
initialize coffee_martini 300
initialize cook_spinach 300
initialize cut_roasted_beef 300
initialize flame_salmon_1 1200
initialize flame_steak 300
initialize sear_steak 300

before_initialize_meetingroom() {
    for i in $(seq 1 $2); do
        mv data/$1/frame$i data/$1/tmpframe$i
        mkdir -p data/$1/frame$i
        mv data/$1/tmpframe$i/input data/$1/frame$i/input
        rm -rf data/$1/tmpframe$i
    done
}
MODE=meetingroom
initialize discussion 300
initialize stepin 300
initialize trimming 300
initialize vrheadset 300

before_initialize_stnerf() {
    for i in $(seq 1 $2); do
        rm -rf data/$1/frame$i
    done
    cd data
    unzip -o $1/$1.zip
    cd ../
}
MODE=stnerf
# initialize taekwondo 101 # camera pose data is too bad
# initialize walking 75    # camera pose data is too bad

before_initialize_dynamic3dgs() {
    for i in $(seq 1 $2); do
        rm -rf data/$1/frame$i
    done
}
MODE=dynamic3dgs
initialize basketball 150
initialize boxes 150
initialize football 150
initialize juggle 150
initialize softball 150
initialize tennis 150

# Dataset without pose
initialize_nopose() {
    eval before_initialize_nopose_$MODE $1 $2 # remove the old data
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-sparse \
        -o "colmap_executable='$COLMAP_EXECUTABLE'"
    n=0
    for i in $(seq 2 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer colmap-sparse \
            -o "load_camera='./data/$1/frame1'" \
            -o "colmap_executable='$COLMAP_EXECUTABLE'" \
            --device cuda
        #     --device cpu &
        # n=$(expr $n + 1)
        # if [ $n -eq 16 ]; then
        #     wait
        #     n=0
        # fi
    done
    # wait
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-dense \
        -o "colmap_executable='$COLMAP_EXECUTABLE'" \
        $INITARGS # dense initialization the first frame
    echo Done $MODE $1 $2
}

before_initialize_nopose_stnerf() {
    before_initialize_stnerf $1 $2
    for i in $(seq 1 $2); do
        rm -rf "data/$1/frame$i/labels" "data/$1/frame$i/pointclouds"
        mv "data/$1/frame$i/images" "data/$1/frame$i/input"
    done
}
MODE=stnerf
initialize_nopose boxing 71
initialize_nopose taekwondo 101
initialize_nopose walking 75
