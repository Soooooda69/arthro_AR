while getopts n: flag
do
    case ${flag} in
        n) sequence=${OPTARG};;
    esac
done
rm -rf output/$sequence/depth
rm -rf output/$sequence/depth_video
rm -rf output/$sequence/disp
rm -rf output/$sequence/ply
python utils/depth_process.py --output_folder output --name $sequence 
python models/depth-to-normal-translator/python/converter.py --root_path output/$sequence 

