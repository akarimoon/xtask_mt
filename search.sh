#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=12:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch
export CUDA_VISIBLE_DEVICES=0,1

python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm -e 1 -b 12 --lr 0.0002 -j 8
# python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm
# python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm
# python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm
# python main_cross_cs.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm