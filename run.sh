#!/bin/bash

for i in {1..100}; do
    python train-rl.py > "output_$i.txt" 2>&1
done
