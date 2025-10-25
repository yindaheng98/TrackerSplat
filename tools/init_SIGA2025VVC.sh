#!/bin/bash

extract_SIGA2025VVC() {
    ROOT=data/SIGA2025VVC-Dataset/$1
    rm -rf $ROOT/images
    tar -zxvf $ROOT/images.tar.gz -C $ROOT
    rm -rf $ROOT/masks
    tar -zxvf $ROOT/masks.tar.gz -C $ROOT
    for frame in $(seq 1 $2); do
        rm -rf $ROOT/frame$frame/images
        mkdir -p $ROOT/frame$frame/images
        echo "Organizing frame $frame"
        for view in $(seq 1 $3); do
            if [ ! -e $ROOT/images/$(printf "%02d" $((view-1))) ] || [ ! -e $ROOT/masks/$(printf "%02d" $((view-1))) ]; then
                continue
            fi
            ln $ROOT/images/$(printf "%02d" $((view-1)))/$(printf "%06d" $((frame-1))).jpg $ROOT/frame$frame/images/$(printf "%02d" $((view-1))).jpg
            ln $ROOT/masks/$(printf "%02d" $((view-1)))/$(printf "%06d" $((frame-1))).png $ROOT/frame$frame/images/$(printf "%02d" $((view-1)))_mask.png
        done
    done
}
extract_SIGA2025VVC compression/test/004_1_seq1 300 58 # debug