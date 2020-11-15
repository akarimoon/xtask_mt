#!/bin/bash

python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.001 -e 100 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.01 --gamma 0.01 -e 100 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm

python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.1 --gamma 0.001 -e 100 --scheduler_step_size 40 --tdep_loss L1 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.1 --gamma 0.001 -e 100 --scheduler_step_size 40 --scheduler_gamma 0.1 --tdep_loss L1 --run_only --notqdm

python main_cross_cs.py -b 8 --label_smoothing 0.1 --uncertainty_weights --alpha 0.1 --gamma 0.001 -e 100 --scheduler_step_size 60 --tdep_loss L1 --run_only --notqdm
