#!/bin/bash

# check effect of temp
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.01 -e 200 --scheduler_step_size 60 --temp 5 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.05 --gamma 0.05 -e 200 --scheduler_step_size 60 --temp 5 --run_only --notqdm

# check effect of batch normalization
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.01 -e 200 --scheduler_step_size 60 --run_only --notqdm --batch_norm

# compare with other alpha/gamma
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.1 --gamma 0.001 -e 200 --scheduler_step_size 60 --run_only --notqdm
