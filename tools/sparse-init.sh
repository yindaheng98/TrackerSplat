#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)
MODE=n3dv
initialize() {
    # echo \
    python tools/parse_camera.py \
        --mode $MODE \
        --path data/$1 \
        --colmap_executable $COLMAP_EXECUTABLE \
        --n_frames $2
    n=0
    for i in $(seq 1 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer colmap-sparse \
            -o "colmap_executable='$COLMAP_EXECUTABLE'" \
            --device cpu &
        n=$(expr $n + 1)
        if [ $n -eq 16 ]; then
            wait
            n=0
        fi
    done
}

MODE=n3dv
initialize coffee_martini 300
initialize cook_spinach 300
initialize cut_roasted_beef 300
initialize flame_salmon_1 1200
initialize flame_steak 300
initialize sear_steak 300

MODE=meetingroom
initialize discussion 300
initialize stepin 300
initialize trimming 300
initialize vrheadset 300

MODE=stnerf
initialize boxing 71
initialize taekwondo 101
initialize walking 75

MODE=dynamic3dgs
initialize_dynamic3dgs basketball 150
initialize_dynamic3dgs boxes 150
initialize_dynamic3dgs football 150
initialize_dynamic3dgs juggle 150
initialize_dynamic3dgs softball 150
initialize_dynamic3dgs tennis 150
