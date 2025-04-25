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
initialize walking 1 # debug
