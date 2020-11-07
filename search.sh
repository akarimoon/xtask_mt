#!/bin/bash

python main_cross.py --alpha 0.4 --gamma 0.0001 --lp logL1 --label_smoothing 0.1 --uncertainty_weights --grad_loss --run_only
python main_cross.py --alpha 0.4 --gamma 0.0001 --lp L1 --label_smoothing 0.1 --uncertainty_weights --grad_loss --run_only
python main_cross.py --alpha 0.4 --gamma 0.00001 --label_smoothing 0.1 --uncertainty_weights --grad_loss --run_only
