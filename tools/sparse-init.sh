#!/bin/bash
initialize() {
    # echo \
    python -m instantsplat.initialize \
        -d data/$1/frame1 \
        --initializer colmap-sparse \
        -o "colmap_executable='$(which colmap)'"
    # -o "colmap_executable='./data/colmap/COLMAP.bat'"
    for i in $(seq 2 $2); do
        # echo \
        python -m instantsplat.initialize \
            -d data/$1/frame$i \
            --initializer colmap-sparse \
            -o "load_camera='./data/$1/frame1'" \
            -o "colmap_executable='$(which colmap)'"
        # -o "colmap_executable='./data/colmap/COLMAP.bat'"
    done
}

# Meeting room datasets
# initialize "stepin" 300
initialize "walking" 75

# Dynamic 3DGS datasets
initialize "basketball" 150
initialize "boxes" 150
initialize "football" 150
initialize "juggle" 150
initialize "softball" 150
initialize "tennis" 150
