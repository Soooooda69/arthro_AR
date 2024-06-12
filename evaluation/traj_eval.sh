while getopts p:s: flag
do
    case ${flag} in
        p) path=${OPTARG};;
        s) save_path=${OPTARG};;
    esac
done

echo "$path"
evo_ape tum "$path"/dataset/arthro_poses_gt.txt "$path"/poses_pred.txt -as --plot -v #--save_results res.zip
# evo_ape tum "$path"/dataset/arthro_poses_gt.txt "$save_path"/zed_arthro_poses.txt -as --plot -v
# evo_traj tum --ref "$path"/dataset/arthro_poses_gt.txt "$save_path"/zed_arthro_poses.txt "$path"/poses_pred.txt -as --plot -v
# evo_ape tum "$path"/zed_poses_gt.txt "$path"/zed_poses.txt -a --plot -v #--save_results res.zip
# evo_traj tum "$path"/zed_poses.txt --plot -v