#!/bin/bash

for i in 0.2 0.5 0.7
do
    for j in 0.2 0.5 0.7
    do
        python main_cross.py -b 6 --alpha "${i}" --gamma "${j}" --label_smoothing 0.1
    done
done