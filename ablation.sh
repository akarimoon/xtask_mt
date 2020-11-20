#!/bin/bash

# (i)
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.0 -e 250 --scheduler_step_size 60 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --gamma 0.0 -e 250 --scheduler_step_size 60 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.0 --gamma 0.0 -e 250 --scheduler_step_size 60 --run_only --notqdm
