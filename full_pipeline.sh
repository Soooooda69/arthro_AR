#!/bin/bash
while getopts n: flag
do
    case ${flag} in
        n) sequence=${OPTARG};;
    esac
done

folders=("left_ForwardBackward" "left_ForwardBackward_2" "left_Pivoting" "left_CommonMotion" "right_CommonMotion" "right_ForwardBackward" "right_Pivoting" "right_Pivoting_2" "surgeon_test_1" "surgeon_test_2")


############################ Running SLAM ################################
rm -r data/temp_data/localize_tracking/*
cd OneSLAM_Arthro
output_folder="../output"
# trial=$sequence
data_root="../data/phantom_data_64/$sequence"
python slam.py --data_root $data_root --output_folder $output_folder --name $sequence --process_subset --start_idx 250 --end_idx 300 \
                    --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 800 \

python ../utils/helper.py --track_path ../data/temp_data/localize_tracking --save_track
cd ..
############################ Depth processing ############################
rm -rf output/$sequence/depth
rm -rf output/$sequence/depth_video
rm -rf output/$sequence/disp
rm -rf output/$sequence/ply
python utils/depth_process.py --output_folder output --name $sequence 
python models/depth-to-normal-translator/python/converter.py --root_path output/$sequence 

############################ GS Surfel reconsturction ############################
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gaussian_surfels
cd models/gaussian_surfels
python train.py -s ../../output/$sequence/colmap_data/ --model_path ../../output/$sequence/gs_surfel
