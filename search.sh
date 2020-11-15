#!/bin/bash

python main_cross_cs.py -n 7 --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.001 -e 200 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm
python main_cross_cs.py -n 7 --label_smoothing 0.0 --uncertainty_weights --alpha 0.01 --gamma 0.001 -e 60 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm
python main_cross_cs.py -n 7 --label_smoothing 0.5 --uncertainty_weights --alpha 0.01 --gamma 0.001 -e 60 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm