#!/bin/bash
while getopts n: flag
do
    case ${flag} in
        n) name=${OPTARG};;
    esac
done

cd evaluation
python reconstruction_eval.py --target ../data/phantom_data_64/knee_64_crop_right.ply --source ../$name/gs_surfel/point_cloud/iteration_10000/point_cloud.ply --save ../$name