#!/bin/bash

python const_energy.py -e 100 --use_xtc --notqdm
python const_energy.py -e 100 --notqdm
python const_energy.py --infer_only