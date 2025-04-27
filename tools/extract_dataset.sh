# !/bin/bash

extract_n3dv() {
    HERE=$(pwd)
    rm -rf "$HERE/data/$1" && cd "$HERE/data" && unzip -o $1.zip && cd "$HERE"
    # echo \
    python tools/extract_n3dv.py \
        --path data/$1 \
        --exec ./data/ffmpeg \
        --fmt "$3" \
        --n_frames $2 >./temp.sh &&
        chmod +x ./temp.sh && ./temp.sh && rm ./temp.sh
}

extract_n3dv coffee_martini 300 "cam[0-9][012456789].mp4" # no cam03, cam13 is wrong
rm data/coffee_martini/cam13.mp4                          # cam13 is wrong
mv data/coffee_martini/poses_bounds.npy data/coffee_martini/poses_bounds_raw.npy
python -c "import numpy as np; poses = np.load('data/coffee_martini/poses_bounds_raw.npy'); np.save('data/coffee_martini/poses_bounds.npy', np.concat((poses[:12, :], poses[13:, :])))"
extract_n3dv cook_spinach 300 "cam[0-9][0-9].mp4"
extract_n3dv cut_roasted_beef 300 "cam[0-9][0-9].mp4"
zip -s- data/flame_salmon_1_split.zip -O data/flame_salmon_1.zip
extract_n3dv flame_salmon_1 1200 "cam[0-9][0-9].mp4"
rm flame_salmon_1.zip
extract_n3dv flame_steak 300 "cam[0-9][0-9].mp4"
extract_n3dv sear_steak 300 "cam[0-9][0-9].mp4"

extract_n3dv discussion 300 "cam_[0-9]+.mp4"
extract_n3dv stepin 300 "cam_[0-9]+.mp4"
extract_n3dv trimming 300 "cam_[0-9]+.mp4"
extract_n3dv vrheadset 300 "cam_[0-9]+.mp4"

extract_dynamic3dgs() {
    HERE=$(pwd)
    rm -rf "$HERE/data/basketball"
    rm -rf "$HERE/data/boxes"
    rm -rf "$HERE/data/football"
    rm -rf "$HERE/data/juggle"
    rm -rf "$HERE/data/softball"
    rm -rf "$HERE/data/tennis"
    unzip -o data/data.zip
}

extract_dynamic3dgs
