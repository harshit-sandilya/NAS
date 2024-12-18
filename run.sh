#!/bin/bash

for i in {1..10}; do
    python train-rl.py --mode $((i % 2)+1) > output$i.log 2>&1
done
