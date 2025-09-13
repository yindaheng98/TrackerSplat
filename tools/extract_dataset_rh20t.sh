# !/bin/bash

extract_n3dv() {
    rm -rf data/$1/frame*
    # echo \
    python tools/extract_rh20t.py \
        --path data/$1 \
        --exec ./data/ffmpeg \
        --fmt "$3" \
        --n_frames $2 >./temp.sh &&
        chmod +x ./temp.sh && ./temp.sh && rm ./temp.sh
}

extract_n3dv RH20T_cfg3/task_0001_user_0016_scene_0002_cfg_0003 300 "cam_[a-z0-9]+[/]color.mp4"
