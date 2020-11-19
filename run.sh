#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=1:ompthreads=1
#PBS -l walltime=10:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch
python main_cross_nyu.py --label_smoothing 0.1 --uncertainty_weights --run_only --notqdm -j 32
