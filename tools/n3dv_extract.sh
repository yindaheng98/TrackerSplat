# !/bin/bash

mkdir -p data/cameras
wget -O data/cameras/cameras.zip https://github.com/yindaheng98/InstantSplatStream/releases/download/v0.0-camera/cameras.zip
cd data/cameras
unzip -o cameras.zip
cd ../../

convert_n3dv() {
    # echo \
    python tools/n3dv_extract.py \
        --path data/$1 \
        --exec ./data/ffmpeg \
        --fmt "$3" \
        --n_frames $2 >./temp.sh &&
        chmod +x ./temp.sh && ./temp.sh && rm ./temp.sh
    # extract first camera from dataset
    rm -rf "data/$1/frame1" && cp -r "data/cameras/$1/frame1" "data/$1/frame1"
}

convert_n3dv coffee_martini 300 "cam[0-9][0-9].mp4"
convert_n3dv cook_spinach 300 "cam[0-9][0-9].mp4"
convert_n3dv cut_roasted_beef 300 "cam[0-9][0-9].mp4"
convert_n3dv flame_salmon_1 1200 "cam[0-9][0-9].mp4"
convert_n3dv flame_steak 300 "cam[0-9][0-9].mp4"
convert_n3dv sear_steak 300 "cam[0-9][0-9].mp4"

convert_n3dv discussion 300 "cam_[0-9]+.mp4"
convert_n3dv stepin 300 "cam_[0-9]+.mp4"
convert_n3dv trimming 300 "cam_[0-9]+.mp4"
convert_n3dv vrheadset 300 "cam_[0-9]+.mp4"

convert_stnerf() {
    for ((i = 1; i <= $2; ++i)); do
        if [ ! -e "data/$1/frame$i/input" ]; then
            rm -rf "data/$1/frame$i/labels" "data/$1/frame$i/pointclouds"
            mv "data/$1/frame$i/images" "data/$1/frame$i/input"
        fi
    done
    # extract first camera from dataset
    rm -rf "data/$1/frame1" && cp -r "data/cameras/$1/frame1" "data/$1/frame1"
}

convert_stnerf boxing 71
convert_stnerf taekwondo 101
convert_stnerf walking 75
