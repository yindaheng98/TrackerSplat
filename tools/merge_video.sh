#!/bin/bash
FFMPEG="ffmpeg"
merge() {
    python tools/merge_video.py output $1 $2 $3 output/collected_videos
    $FFMPEG -y -f image2 -framerate 15 -i output/collected_videos/$1/%05d.png -vcodec libx264 -pix_fmt yuv420p -crf 22 -preset veryslow output/collected_videos/$1.mp4
}
merge walking st-nerf 75
merge taekwondo st-nerf 101
merge boxing st-nerf 71

merge coffee_martini N3DV 300
merge cook_spinach N3DV 300
merge cut_roasted_beef N3DV 300
merge flame_salmon_1 N3DV 1200
merge flame_steak N3DV 300
merge sear_steak N3DV 300

merge discussion MeetingRoom 300
merge stepin MeetingRoom 300
merge trimming MeetingRoom 300
merge vrheadset MeetingRoom 300

merge basketball Dynamic3DGS 150
merge boxes Dynamic3DGS 150
merge football Dynamic3DGS 150
merge juggle Dynamic3DGS 150
merge softball Dynamic3DGS 150
merge tennis Dynamic3DGS 150

$FFMPEG -f concat -safe 0 -i tools/videos.txt -c copy videos.mp4