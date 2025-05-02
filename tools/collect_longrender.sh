#!/bin/bash
FFMPEG="ffmpeg"
collect() {
    mkdir -p $(dirname output/collected_videos/$1/$2.mp4)
    cp output/$1/$2.mp4 output/collected_videos/$1/$2.mp4
}
collect_all() {
    collect $1 refine/base-propagate-dot-cotracker3 $2 $3 $4 $5
    collect $1 refine/base-base-dot-cotracker3 $2 $3 $4 $5
    collect $1 train/regularized $2 $3 $4 $5
    collect $1 train/base $2 $3 $4 $5
    collect $1 train/hexplane $2 $3 $4 $5
    collect $1 train/regularizedhexplane $2 $3 $4 $5
}
collect_all walking 10000 1000 2 75
collect_all taekwondo 10000 1000 2 101
collect_all boxing 10000 1000 2 71

collect_all coffee_martini 10000 1000 2 300
collect_all cook_spinach 10000 1000 2 300
collect_all cut_roasted_beef 10000 1000 2 300
collect_all flame_salmon_1 10000 1000 2 1200
collect_all flame_steak 10000 1000 2 300
collect_all sear_steak 10000 1000 2 300

collect_all discussion 10000 1000 2 300
collect_all stepin 10000 1000 2 300
collect_all trimming 10000 1000 2 300
collect_all vrheadset 10000 1000 2 300

collect_all basketball 10000 1000 2 150
collect_all boxes 10000 1000 2 150
collect_all football 10000 1000 2 150
collect_all juggle 10000 1000 2 150
collect_all softball 10000 1000 2 150
collect_all tennis 10000 1000 2 150
