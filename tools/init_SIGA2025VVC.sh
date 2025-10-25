#!/bin/bash

extract_SIGA2025VVC() {
    mkdir -p data/SIGA2025VVC-Dataset/$1/images
    tar -zxvf data/SIGA2025VVC-Dataset/$1/images.tar.gz -C data/SIGA2025VVC-Dataset/$1
    mkdir -p data/SIGA2025VVC-Dataset/$1/masks
    tar -zxvf data/SIGA2025VVC-Dataset/$1/masks.tar.gz -C data/SIGA2025VVC-Dataset/$1
}
extract_SIGA2025VVC compression/test/004_1_seq1 # debug