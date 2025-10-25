#!/bin/bash

extract_SIGA2025VVC() {
    ROOT=data/SIGA2025VVC-Dataset/$1
    rm -rf $ROOT/images
    tar -zxvf $ROOT/images.tar.gz -C $ROOT
    rm -rf $ROOT/masks
    tar -zxvf $ROOT/masks.tar.gz -C $ROOT
    for frame in $(seq 0 $2); do
        rm -rf $ROOT/frame$((frame + 1))
        mkdir -p $ROOT/frame$((frame + 1))/images
        for view in $(seq 0 $3); do
            mv $ROOT/images/$(printf "%02d" $view)/$(printf "%06d" $frame).jpg $ROOT/frame$((frame + 1))/images/$(printf "%02d" $view).jpg
            mv $ROOT/masks/$(printf "%02d" $view)/$(printf "%06d" $frame).png $ROOT/frame$((frame + 1))/images/$(printf "%02d" $view)_mask.png
        done
    done
}
extract_SIGA2025VVC compression/test/004_1_seq1 300 57 # debug