#!/bin/bash
#PBS -q h-regular
#PBS -W group_list=po9025
#PBS -l select=1:mpiprocs=4
#PBS -l walltime=35:00:00
cd $PBS_O_WORKDIR
source /lustre/po9025/o09025/.bashrc
conda activate pytorch
# export CUDA_VISIBLE_DEVICES=0,1 

# python main_cross_cs.py --uncertainty_weights --run_only --notqdm -j 8 --multiple_gpu
# python main_cross_cs.py --uncertainty_weights --label_smoothing 0.1 --run_only --notqdm -j 8 --multiple_gpu
# python main_cross_cs.py --uncertainty_weights --run_only --notqdm -j 8 --multiple_gpu --alpha 0.0 --gamma 0.0 --multiple_gpu
# python main_cross_cs.py --uncertainty_weights --label_smoothing 0.1 --run_only --notqdm -j 8 --multiple_gpu --alpha 0.0 --gamma 0.0 --multiple_gpu

# python model/mtan/model_segnet_mtan_cs.py --dataroot data/cityscapes/ --weight uncert --pcgrad
# python model/mtan/model_segnet_mtan_nyu.py --dataroot data/nyu_preprocessed/ --weight uncert --pcgrad

python main_cross_nyu.py --uncertainty_weights --label_smoothing 0.1 --run_only --notqdm -b 2 --lr 0.00005