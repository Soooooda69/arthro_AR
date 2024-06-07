#!/bin/bash
while getopts n: flag
do
    case ${flag} in
        n) name=${OPTARG};;
    esac
done

source ~/anaconda3/etc/profile.d/conda.sh
conda activate gaussian_surfels
cd models/gaussian_surfels
python train.py -s ../../output/$name/colmap_data/ --model_path ../../output/$name/gs_surfel

# python render.py -m ../../output/$name/gs --img --depth 10