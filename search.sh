#!/bin/bash

python main_cross_cs.py --uncertainty_weights --label_smoothing 0.1 --run_only --notqdm --optim sgd
python main_cross_cs.py --run_only --notqdm