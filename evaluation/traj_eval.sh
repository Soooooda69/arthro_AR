while getopts p:s: flag
do
    case ${flag} in
        p) path=${OPTARG};;
        s) save_path=${OPTARG};;
    esac
done

echo "$path"
evo_ape tum "$path"/dataset/arthro_poses_gt.txt "$path"/poses_pred.txt -as --plot -v #--save_results res.zip
