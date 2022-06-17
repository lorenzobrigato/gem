#!/bin/bash


# DEFAULT CONFIG
DS=isic2018
ARCH=rn50
EPOCHS_HPO=500
TRIALS_HPO=100
GP=25
TRIALS_FTRAIN=10


for i in 0 1 2

do

    echo "Running experiments on split ${i}"
    
    python scripts/full_train.py ${DS} \
        --architecture $ARCH \
        --target-size 224 \
        --min-scale 0.4 \
        --vflip \
        --batch-sizes 8 16 32 \
        --epochs $EPOCHS_HPO \
        --grace-period $GP \
        --num-trials $TRIALS_HPO \
        --num-trials-f $TRIALS_FTRAIN \
        --train-split train${i} --test-split val${i} --train-split-f trainval${i} --test-split-f test${i} \
        "$@"

done
