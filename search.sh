#!/bin/bash

python const_energy.py -e 100 --use_xtc
python const_energy.py -e 100
python const_energy.py --infer_only