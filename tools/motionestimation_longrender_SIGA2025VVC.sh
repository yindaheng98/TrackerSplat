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
}
render_all SIGA2025VVC-Dataset/compression/test/004_1_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/test/006_1_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/test/007_0_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/test/008_2_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/test/008_2_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/test/011_0_seq1 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/val/001_1_seq0 10000 1000 2 300
render_all SIGA2025VVC-Dataset/compression/val/012_0_seq0 10000 1000 2 300
