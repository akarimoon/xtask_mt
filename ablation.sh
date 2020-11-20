#!/bin/bash

# (i)
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.0 --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --gamma 0.0  --run_only --notqdm
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --alpha 0.0 --gamma 0.0 --run_only --notqdm

# (ii)
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --use_pretrain
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --enc_layers 50
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --enc_layers 101

# (iii)
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --temp 5
python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --temp 5 --alpha 0.05 --gamma 0.05
