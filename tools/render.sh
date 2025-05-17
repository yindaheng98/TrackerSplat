#!/bin/bash
rendering() {
    # echo \
    python -m instantsplatstream.incrementalrender \
        -s data/$1/frame$4 \
        --destination_base output/$1/frame$3 \
        -d output/$1/$2/frame$4 \
        --iteration_base 10000 -i 1000
}
renderall() {
    rendering $1 "refine/base-propagate-dot-cotracker3" $2 $3
    # rendering $1 "refine/base-base-dot-cotracker3" $2 $3
    rendering $1 "train/hicom" $2 $3
    rendering $1 "train/regularized" $2 $3
    rendering $1 "train/hexplane" $2 $3
    # rendering $1 "train/regularizedhexplane" $2 $3
}
renderall "walking" 61 69
renderall "stepin" 61 69
renderall "basketball" 1 9
# renderall "tennis" 1 9
renderall "softball" 141 149
renderall "taekwondo" 81 89
renderall "coffee_martini" 1 9