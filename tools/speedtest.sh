#!/bin/bash
INITTRAININGITERS=10000
speedtest() {
    PARALLELDEVICES=""
    for i in $(seq 0 $(expr $6 - 1)); do
        PARALLELDEVICES="$PARALLELDEVICES --parallel_device=$i"
    done
    # echo \
    python -m instantsplatstream.motionestimation_speedtest \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $3 \
        --pipeline $4 $5 \
        -n $6 -b $(expr $6 + 1) \
        $PARALLELDEVICES
}
# speedtest "discussion" 61 1000 track/propagate-dot-cotracker3 "-o rescale_factor=0.3" 4 # debug
speedtest_allmethods() {
    speedtest $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5
    speedtest $1 $2 $3 refine/base-base-dot-cotracker3 "-o rescale_factor=$4" $5
    speedtest $1 $2 $3 train/regularized "-o neighbors=20" $5
    speedtest $1 $2 $3 train/base "" $5
    speedtest $1 $2 $3 train/hexplane "" $5
    speedtest $1 $2 $3 train/regularizedhexplane "-o neighbors=20" $5
}

DEVICES=4

FRAME=51
speedtest_allmethods "walking" $FRAME 1000 0.3 $DEVICES
speedtest_allmethods "taekwondo" $FRAME 1000 0.3 $DEVICES

speedtest_allmethods "coffee_martini" $FRAME 1000 0.3 $DEVICES
speedtest_allmethods "cook_spinach" $FRAME 1000 0.3 $DEVICES
speedtest_allmethods "cut_roasted_beef" $FRAME 1000 0.3 $DEVICES

INITARGS="-o use_fused=True"
speedtest_allmethods "discussion" $FRAME 1000 0.5 $DEVICES
speedtest_allmethods "stepin" $FRAME 1000 0.5 $DEVICES
speedtest_allmethods "trimming" $FRAME 1000 0.5 $DEVICES
speedtest_allmethods "vrheadset" $FRAME 1000 0.5 $DEVICES

INITARGS="-o use_fused=True"
speedtest_allmethods "basketball" $FRAME 1000 1.0 $DEVICES
speedtest_allmethods "boxes" $FRAME 1000 1.0 $DEVICES
speedtest_allmethods "football" $FRAME 1000 1.0 $DEVICES
speedtest_allmethods "juggle" $FRAME 1000 1.0 $DEVICES
speedtest_allmethods "softball" $FRAME 1000 1.0 $DEVICES
speedtest_allmethods "tennis" $FRAME 1000 1.0 $DEVICES
