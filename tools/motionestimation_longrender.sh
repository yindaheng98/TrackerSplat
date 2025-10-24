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
    python -m trackersplat.render \
        -d output/$1/$2 \
        --data_dir output/$1/$2/render \
        --destination_init output/$1/frame1 \
        --iteration_init $3 -i $4 \
        --frame_start $5 --frame_end $6 \
        --load_camera output/$1/frame1/cameras.json \
        --mode base
    $FFMPEG -y -f image2 -i output/$1/$2/render/%05d.png -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p -crf 10 output/$1/$2.mp4
    rm -rf output/$1/$2/render
}
render_all() {
    render $1 refine/base-propagate-dot-cotracker3 $2 $3 $4 $5
    render $1 refine/base-base-dot-cotracker3 $2 $3 $4 $5
    render $1 train/regularized $2 $3 $4 $5
    render $1 train/hicom $2 $3 $4 $5
    render $1 train/hexplane $2 $3 $4 $5
    render $1 train/regularizedhexplane $2 $3 $4 $5
}
render_all walking 10000 1000 2 75
render_all taekwondo 10000 1000 2 101
render_all boxing 10000 1000 2 71

render_all coffee_martini 10000 1000 2 300
render_all cook_spinach 10000 1000 2 300
render_all cut_roasted_beef 10000 1000 2 300
render_all flame_salmon_1 10000 1000 2 1200
render_all flame_steak 10000 1000 2 300
render_all sear_steak 10000 1000 2 300

render_all discussion 10000 1000 2 300
render_all stepin 10000 1000 2 300
render_all trimming 10000 1000 2 300
render_all vrheadset 10000 1000 2 300

render_all basketball 10000 1000 2 1500
render_all boxes 10000 1000 2 1500
render_all football 10000 1000 2 1500
render_all juggle 10000 1000 2 1500
render_all softball 10000 1000 2 1500
render_all tennis 10000 1000 2 1500
