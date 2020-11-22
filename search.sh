#!/bin/bash

python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.01 -e 250 --scheduler_step_size 60 --run_only --notqdm
python main_cross_cs.py --uncertainty_weights --alpha 0.01 --gamma 0.01 -e 250 --scheduler_step_size 60 --run_only --notqdm
python main_cross_cs.py　--run_only --notqdm