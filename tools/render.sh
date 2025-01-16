#!/bin/bash
rendering() {
    # echo \
    python -m instantsplatstream.incrementalrender \
        -s data/$1/frame$4 \
        --destination_base output/$1/frame$3 \
        -d output/$1/$2/frame$4 \
        --iteration_base 10000 -i 1000
}
rendering "walking" "train/regularizedhexplane" 61 68
