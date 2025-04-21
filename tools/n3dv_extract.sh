# !/bin/bash

mkdir -p data/saved_frame1
wget -O data/saved_frame1/saved_frame1.zip https://github.com/yindaheng98/InstantSplatStream/releases/download/v0.0-camera/saved_frame1.zip
cd data/saved_frame1
unzip -o saved_frame1.zip
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
    rm -rf "data/$1/frame1" && cp -r "data/saved_frame1/$1/frame1" "data/$1/frame1"
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
