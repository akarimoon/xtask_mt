#!/bin/bash

python main_cross_cs.py -n 7 --label_smoothing 0.1 --uncertainty_weights --scheduler_gamma 0.5 --height 128 --width 256 --is_shallow -e 40 --run_only --infer_only --exp_num 17
python main_cross_cs.py -n 7 --gamma 0.001 --label_smoothing 0.1 --uncertainty_weights --scheduler_gamma 0.5 --height 128 --width 256 --is_shallow -e 40 --run_only --infer_only --exp_num 18
python main_cross_cs.py -n 7 --gamma 0.00005 --label_smoothing 0.1 --uncertainty_weights --scheduler_gamma 0.5 --height 128 --width 256 --is_shallow -e 40 --run_only --infer_only --exp_num 19
python main_cross_cs.py -n 7 --alpha 0.2 --label_smoothing 0.1 --uncertainty_weights --scheduler_gamma 0.5 --height 128 --width 256 --is_shallow -e 40 --run_only --infer_only --exp_num 20
