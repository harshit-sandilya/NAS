#!/bin/bash

for i in {2..10}; do
    python train-rl.py --mode $((i % 2)) > output$i.log 2>&1
done
