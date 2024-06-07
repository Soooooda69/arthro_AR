while getopts n:s: flag
do
    case ${flag} in
        n) data=${OPTARG};;
        s) sequence=${OPTARG};;
    esac
done


# mkdir -p data/$data/$sequence/raw_images
# mkdir -p data/$data/$sequence/raw_poses
# mv data/$data/$sequence/*.jpg data/$data/$sequence/raw_images
# mv data/$data/$sequence/*.json data/$data/$sequence/raw_poses
# python utils/dataset.py --root_dir data/$data/$sequence --calib_dir data/$data --downsample

# folders=("left_Pivoting" "left_CommonMotion" "left_ForwardBackward" "left_ForwardBackward_2" "right_CommonMotion" "right_ForwardBackward" "right_Pivoting" "right_Pivoting_2" "surgeon_test_1" "surgeon_test_2")
# folders=("left_CommonMotion")
folders=("left_p1" "left_p2" "left_cm" "left_r" "left_fb" "right_p1" "right_p2" "right_cm" "right_r" "right_fb")

for folder in "${folders[@]}"; do
    mkdir -p data/$data/$folder/raw_images
    mkdir -p data/$data/$folder/raw_poses
    mv data/$data/$folder/*.jpg data/$data/$folder/raw_images
    mv data/$data/$folder/*.json data/$data/$folder/raw_poses
    python utils/dataset.py --root_dir data/$data/$folder --calib_dir data/$data --downsample
done
