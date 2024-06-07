#!/bin/bash
while getopts n: flag
do
    case ${flag} in
        n) name=${OPTARG};;
    esac
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate sugar
cd models/GaussianObject_o
python train_gs.py -s ../../output/$name/colmap_data/ --model_path ../../output/$name/gs
#  python train_gs.py -s data/data_h_200_scaledDepth/colmap_data/