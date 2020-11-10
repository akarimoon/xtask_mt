#!/bin/bash

python main_cross.py -n 7 --alpha 0.4 --gamma 0.0001 --lp L1 --label_smoothing 0.1 --uncertainty_weights --scheduler_gamma 0.5 --run_only
python main_cross.py -n 7 --alpha 0.4 --gamma 0.0001 --lp logL1 --label_smoothing 0.1 --scheduler_gamma 0.5 --run_only
python main_cross.py -n 7 --alpha 0.4 --gamma 0.0001 --lp logL1 --label_smoothing 0.1 --uncertainty_weights --grad_loss --scheduler_gamma 0.5 --run_only