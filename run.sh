#!/bin/bash

for i in {1..10}; do
    python train-rl.py > output$i.log 2>&1
done
