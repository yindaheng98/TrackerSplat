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
speedtest "discussion" 61 1000 track/propagate-dot-cotracker3 "-o rescale_factor=0.3" 4 # debug
