#!/bin/bash


# DEFAULT CONFIG
DS=cub
ARCH=rn50
EPOCHS_HPO=200
TRIALS_HPO=100
GP=10
TRIALS_FTRAIN=10


for i in 0 1 2

do

    echo "Running experiments on split ${i}"

    python scripts/full_train.py ${DS} \
        --architecture $ARCH \
        --target-size 224 \
        --min-scale 0.4 \
        --batch-sizes 8 16 32 \
        --epochs $EPOCHS_HPO \
        --grace-period $GP \
        --num-trials $TRIALS_HPO \
        --train-split train${i} --test-split val${i} --train-split-f trainval${i} \
        "$@"

done
