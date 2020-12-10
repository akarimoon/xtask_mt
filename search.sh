#!/bin/bash

python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --alpha 0.0 --gamma 0.0
python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --label_smoothing 0.1 --alpha 0.0 --gamma 0.0

python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --use_pretrained
python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --enc_layers 50
python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --enc_layers 101

python main_cross_cs.py --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm

python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --batch_norm
python main_cross_cs.py --uncertainty_weights --label_smoothing 0.1 --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --batch_norm
python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --wider_ttnet
python main_cross_cs.py --uncertainty_weights --run_only -b 8 -e 250 --scheduler_step_size 80 --notqdm --wider_ttnet --batch_norm