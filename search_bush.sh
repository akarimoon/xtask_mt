#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=40:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch
export CUDA_VISIBLE_DEVICES=0,1 

# python model/mtan/model_segnet_mtan_cs.py --dataroot data/cityscapes/ --weight dwa
# python model/mtan/model_segnet_mtan_nyu.py --dataroot data/nyu/ --weight dwa -n 2
# python model/mtan/model_segnet_mtan_nyu.py --dataroot data/nyu/ --weight dwa -n 3

# python main_cross_nyu.py --run_only --notqdm --uncertainty_weights --label_smoothing 0.1


# 2470392
python main_cross_cs.py --run_only --notqdm --multiple_gpu --uncertainty_weights -n 19
python main_cross_cs.py --run_only --notqdm --multiple_gpu --uncertainty_weights -n 19 --alpha 0.0 --gamma 0.0