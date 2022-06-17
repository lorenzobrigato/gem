#!/bin/bash


# DEFAULT CONFIG
DS=cifair10
ARCH=wrn-16-8
EPOCHS_HPO=500
TRIALS_HPO=250
GP=50
TRIALS_FTRAIN=10


for i in 0 1 2

do

    echo "Running experiments on split ${i}"

    python scripts/full_train.py ${DS} \
        --architecture $ARCH \
        --rand-shift 4 \
        --epochs $EPOCHS_HPO \
        --grace-period $GP \
        --num-trials $TRIALS_HPO \
        --num-trials-f $TRIALS_FTRAIN \
        --train-split train${i} --test-split val${i} --train-split-f trainval${i} \
        "$@"

done

