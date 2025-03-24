#!/bin/bash
# COLMAP_EXECUTABLE=./data/colmap/COLMAP.bat
COLMAP_EXECUTABLE=$(which colmap)
INITARGS=""
initialize() {
    echo data/$1/frame1
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-dense \
        -o "colmap_executable='$COLMAP_EXECUTABLE'" $INITARGS
}

initialize coffee_martini
initialize cook_spinach
initialize cut_roasted_beef
initialize flame_salmon_1
initialize flame_steak
initialize sear_steak

initialize boxing
initialize taekwondo
initialize walking

INITARGS="-o use_fused=True"

initialize discussion
initialize stepin
initialize trimming
initialize vrheadset

initialize basketball
initialize boxes
initialize football
initialize juggle
initialize softball
initialize tennis
