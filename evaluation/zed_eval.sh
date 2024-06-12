while getopts p: flag
do
    case ${flag} in
        p) root_path=${OPTARG};;
    esac
done

folders=("left_p1" "left_p2" "left_cm" "left_r" "left_fb" "right_p1" "right_p2" "right_cm" "right_r" "right_fb")

for file in "${folders[@]}"; do
    path="$root_path""$file"
    echo "$path"
    mkdir -p "$path"/zed_eval_results
    # evo_ape tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a --plot -v #--save_results res.zip

    # evo_ape tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -as --plot
    evo_ape tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a > "$path"/zed_eval_results/APE_trans.txt
    evo_ape tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a --pose_relation angle_deg > "$path"/zed_eval_results/APE_rot.txt
    evo_rpe tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a --delta 1 > "$path"/zed_eval_results/RPE_trans.txt
    evo_rpe tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a --pose_relation angle_deg  --delta 1  > "$path"/zed_eval_results/RPE_rot.txt
done