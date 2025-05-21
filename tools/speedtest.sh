#!/bin/bash
INITTRAININGITERS=10000
speedtest() {
    PARALLELDEVICES=""
    for i in $(seq 0 $(expr $6 - 1)); do
        PARALLELDEVICES="$PARALLELDEVICES --parallel_device=$i"
    done
    echo $1/$4 -n $(expr $6 \* 3) -b $(expr $6 + 1)
    # echo \
    python -m instantsplatstream.motionestimation_speedtest \
        -s data/$1 -d output/$1 --start_frame $2 \
        --iteration_init $INITTRAININGITERS -i $3 \
        --pipeline $4 $5 \
        -n $(expr $6 \* 3) -b $(expr $6 + 1) \
        $PARALLELDEVICES
}
# speedtest "discussion" 61 1000 track/propagate-dot-cotracker3 "-o rescale_factor=0.3" 4 # debug
speedtest_allmethods() {
    speedtest $1 $2 $3 refine/base-propagate-dot-cotracker3 "-o rescale_factor=$4" $5
    speedtest $1 $2 $3 refine/base-base-dot-cotracker3 "-o rescale_factor=$4" $5
    speedtest $1 $2 $3 train/base "" $5
    speedtest $1 $2 $3 train/regularized "-o neighbors=20" $5
    speedtest $1 $2 $3 train/hicom "" $5
    speedtest $1 $2 $3 train/hexplane "" $5
    speedtest $1 $2 $3 train/regularizedhexplane "-o neighbors=20" $5
}

speedtest_allmethods_alldevices() {
    speedtest_allmethods $1 $2 $3 $4 2
    speedtest_allmethods $1 $2 $3 $4 4
    speedtest_allmethods $1 $2 $3 $4 8
}

FRAME=1
speedtest_allmethods_alldevices "walking" $FRAME 1000 0.3
speedtest_allmethods_alldevices "taekwondo" $FRAME 1000 0.3
speedtest_allmethods_alldevices "boxing" $FRAME 1000 0.3

speedtest_allmethods_alldevices "coffee_martini" $FRAME 1000 0.3
speedtest_allmethods_alldevices "cook_spinach" $FRAME 1000 0.3
speedtest_allmethods_alldevices "cut_roasted_beef" $FRAME 1000 0.3

speedtest_allmethods_alldevices "discussion" $FRAME 1000 0.5
speedtest_allmethods_alldevices "stepin" $FRAME 1000 0.5
speedtest_allmethods_alldevices "trimming" $FRAME 1000 0.5
speedtest_allmethods_alldevices "vrheadset" $FRAME 1000 0.5

speedtest_allmethods_alldevices "basketball" $FRAME 1000 1.0
speedtest_allmethods_alldevices "boxes" $FRAME 1000 1.0
speedtest_allmethods_alldevices "football" $FRAME 1000 1.0
speedtest_allmethods_alldevices "juggle" $FRAME 1000 1.0
speedtest_allmethods_alldevices "softball" $FRAME 1000 1.0
speedtest_allmethods_alldevices "tennis" $FRAME 1000 1.0
