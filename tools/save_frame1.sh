# !/bin/bash

save_frame1() {
    echo saving $1
    rm -rf data/saved_frame1/data/$1
    mkdir -p data/saved_frame1/data/$1
    cp -r data/$1/frame1 data/saved_frame1/data/$1/frame1
    rm -rf data/saved_frame1/output/$1
    mkdir -p data/saved_frame1/output/$1
    cp -r output/$1/frame1 data/saved_frame1/output/$1/frame1
}

rm -rf data/saved_frame1
mkdir -p data/saved_frame1

save_frame1 coffee_martini
save_frame1 cook_spinach
save_frame1 cut_roasted_beef
save_frame1 flame_salmon_1
save_frame1 flame_steak
save_frame1 sear_steak

save_frame1 boxing
save_frame1 taekwondo
save_frame1 walking

save_frame1 discussion
save_frame1 stepin
save_frame1 trimming
save_frame1 vrheadset

save_frame1 basketball
save_frame1 boxes
save_frame1 football
save_frame1 juggle
save_frame1 softball
save_frame1 tennis

cd data/saved_frame1
rm saved_frame1.zip
zip -r saved_frame1-data.zip ./data
zip -r saved_frame1-output.zip ./output
