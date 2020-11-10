#!/bin/bash

# compare against MTAN
python main_cross_cs.py -n 7 --height 128 --width 256 --label_smoothing 0.1 --uncertainty_weights --grad_loss --scheduler_step_size 30 -e 70 --is_shallow --run_only
python main_cross_cs.py -n 7 --height 128 --width 256 --label_smoothing 0.1 --uncertainty_weights --grad_loss --scheduler_step_size 30 -e 70 --run_only
# compare against uncert. weights, boosting
python main_cross_cs.py --height 128 --width 256 --label_smoothing 0.1 --uncertainty_weights --grad_loss --scheduler_step_size 30 -e 70 --run_only
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --grad_loss --scheduler_step_size 30 -e 70 --run_only