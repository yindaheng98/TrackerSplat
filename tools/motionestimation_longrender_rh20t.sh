#!/bin/bash
FFMPEG="ffmpeg"
render() {
    for i in $(seq $5 $6); do
        EXISTSPATH="output/$1/$2/frame$i/point_cloud/iteration_$4/point_cloud.ply"
        if [ ! -e "$EXISTSPATH" ]; then
            echo "not exists: $EXISTSPATH"
            return
        fi
    done
    echo "all exists: output/$1/$2/frame<$5-$6>/point_cloud/iteration_$3/point_cloud.ply"
    python -m instantsplatstream.render \
        -d output/$1/$2 \
        --data_dir output/$1/$2/render \
        --destination_init output/$1/frame1 \
        --iteration_init $3 -i $4 \
        --frame_start $5 --frame_end $6 \
        --interp_n $7 \
        --load_camera output/$1/frame1/cameras.json \
        --mode base
    $FFMPEG -y -f image2 -i output/$1/$2/render/%05d.png -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p -crf 10 output/$1/$2.mp4
    rm -rf output/$1/$2/render
}
n_frames() {
    n=0
    while [ -d "data/$1/frame$(expr $n + 1)" ]; do
        n=$(expr $n + 1)
    done
    echo $n
}
render_all() {
    N=$(n_frames $1)
    NI=$(expr $N \* 10)
    render $1 refine/base-propagate-dot-cotracker3 $2 $3 2 $N $NI
    render $1 refine/base-base-dot-cotracker3 $2 $3 2 $N $NI
    render $1 train/regularized $2 $3 2 $N $NI
    render $1 train/hicom $2 $3 2 $N $NI
    render $1 train/hexplane $2 $3 2 $N $NI
    render $1 train/regularizedhexplane $2 $3 2 $N $NI
}
# render_all RH20T_cfg5/task_0001_user_0016_scene_0002_cfg_0005 10000 1000 # debug
for s in output/RH20T_cfg5/task_*_user_*_scene_*_cfg_0005; do
    render_all ${s:7} 10000 1000 # debug
done
