#!/bin/bash

rm -r data/temp_data/localize_tracking/*


cd OneSLAM_Arthro
folders=("left_ForwardBackward" "left_ForwardBackward_2" "left_Pivoting" "left_CommonMotion" "right_CommonMotion" "right_ForwardBackward" "right_Pivoting" "right_Pivoting_2" "surgeon_test_1" "surgeon_test_2")
output_folder="../output"

# trial="left_CommonMotion"
# data_root="../data/phantom_data_521/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 200 --end_idx 400 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 800\

# trial="left_Pivoting"
# data_root="../data/phantom_data_521/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 1 --end_idx 200 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 600\

# trial="left_ForwardBackward"
# data_root="../data/phantom_data_521/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 100 --end_idx 300 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 600\

# trial="right_CommonMotion"
# data_root="../data/phantom_data_521/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 1 --end_idx 200 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 2000\  

# trial="right_Pivoting"
# data_root="../data/phantom_data_521/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 1 --end_idx 200 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 2000\  

#*********************************** 64 ***********************************
# trial="left_P_test"
# data_root="../data/phantom_data_64/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 1 --end_idx 200 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 800\

# trial="left_cm"
# data_root="../data/phantom_data_64/$trial"
# python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 200 --end_idx 400 \
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 800\

trial="right_cm"
data_root="../data/phantom_data_64/$trial"
python slam.py --data_root $data_root --output_folder $output_folder --name $trial --process_subset --start_idx 1 --end_idx 250 \
                    --section_length 6 --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 800 \


python ../utils/helper.py --track_path ../data/temp_data/localize_tracking --save_track



# python run_slam.py --data_root $data_root --output_folder $output_folder --name $trial+"_new" --process_subset --start_idx 1 --end_idx 400 --image_subsample 1\
#                     --local_ba_size 12 --tracked_point_num_min 600 --tracked_point_num_max 600\


