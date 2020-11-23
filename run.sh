#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=12:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch

python main_cross_nyu.py --label_smoothing 0.1 --uncertainty_weights  -j 64 --run_only --notqdm --alpha 0.001 --gamma 0.001 
python main_cross_nyu.py --label_smoothing 0.1 --uncertainty_weights  -j 64 --run_only --notqdm --alpha 0.01 --gamma 0.01
python main_cross_nyu.py --label_smoothing 0.1 --uncertainty_weights  -j 64 --run_only --notqdm --alpha 0.1 --gamma 0.1