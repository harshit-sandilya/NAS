#!/bin/bash

for i in {1..10}; do
    if ((i % 2)); then
        python train-rl.py --mode 1 > output$i.log 2>&1
    else
        python train-rl.py --mode 2 > output$i.log 2>&1
    fi
done
