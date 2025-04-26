# !/bin/bash

extract_n3dv() {
    # echo \
    python tools/extract_n3dv.py \
        --path data/$1 \
        --exec ./data/ffmpeg \
        --fmt "$3" \
        --n_frames $2 >./temp.sh &&
        chmod +x ./temp.sh && ./temp.sh && rm ./temp.sh
}

extract_n3dv coffee_martini 300 "cam[0-9][0-9].mp4"
extract_n3dv cook_spinach 300 "cam[0-9][0-9].mp4"
extract_n3dv cut_roasted_beef 300 "cam[0-9][0-9].mp4"
extract_n3dv flame_salmon_1 1200 "cam[0-9][0-9].mp4"
extract_n3dv flame_steak 300 "cam[0-9][0-9].mp4"
extract_n3dv sear_steak 300 "cam[0-9][0-9].mp4"

extract_n3dv discussion 300 "cam_[0-9]+.mp4"
extract_n3dv stepin 300 "cam_[0-9]+.mp4"
extract_n3dv trimming 300 "cam_[0-9]+.mp4"
extract_n3dv vrheadset 300 "cam_[0-9]+.mp4"
