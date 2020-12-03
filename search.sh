#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=30:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch
export CUDA_VISIBLE_DEVICES=0, 1 

python main_cross_cs.py --uncertainty_weights --run_only --notqdm -e 200 --alpha 0.0 --gamma 0.0 -j 8