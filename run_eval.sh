#!/bin/bash
while getopts n: flag
do
    case ${flag} in
        n) name=${OPTARG};;
    esac
done

cd evaluation
python reconstruction_eval.py --target ../data/phantom_data_64/knee_64_crop.ply --source ../$name