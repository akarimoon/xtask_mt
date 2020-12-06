#!/bin/bash

python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm --wider_ttnet
python main_cross_cs.py --uncertainty_weights --run_only --notqdm --batch_norm --wider_ttnet
