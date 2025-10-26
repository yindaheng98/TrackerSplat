#!/bin/bash

extract_SIGA2025VVC() {
    ROOT=data/SIGA2025VVC-Dataset/$1
    rm -rf $ROOT/images
    tar -zxvf $ROOT/images.tar.gz -C $ROOT
    rm -rf $ROOT/masks
    tar -zxvf $ROOT/masks.tar.gz -C $ROOT
    for frame in $(seq 1 $2); do
        rm -rf $ROOT/frame$frame
    done
    python tools/parse_camera_SIGA2025VVC.py --path $ROOT --n_frames $2
}
extract_SIGA2025VVC compression/test/004_1_seq1 300 # debug