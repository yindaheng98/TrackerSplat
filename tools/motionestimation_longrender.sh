#!/bin/bash
FFMPEG="ffmpeg"
render() {
    python -m cagstream.render \
        -s output/$1/$2 \
        -d output/$1/$2/render \
        --source_init output/$1/frame1 \
        --iteration_init $3 -i $4 \
        --frame_start $5 --frame_end $6 \
        --load_camera output/$1/frame1/cameras.json \
        --mode base --load_camera_mode interp-json \
        --fix_intrinsic $INTRINSIC
    $FFMPEG -y -f image2 -i output/$1/$2/render/%05d.png -vf "crop=trunc(iw/2)*2:trunc(ih/2)*2" -vcodec libx264 -pix_fmt yuv420p -crf 10 output/$1/$2.mp4
    rm -rf output/$1/$2/render
}
render_all() {
    render $1 train/regularized $2 $3 $4 $5
    render $1 refine/masked-propagate-dot-cotracker3 $2 $3 $4 $5
}
render_all basketball 10000 1000 2 75
