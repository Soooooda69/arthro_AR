import os
import cv2
import shutil
import json
import numpy as np
from pathlib import Path
from natsort import natsorted
from scipy.spatial.transform import Rotation as R
import argparse
from tqdm import tqdm

def load_cam_params(cam_param_dir):
    # Load camera parameters
    with open(cam_param_dir / 'calibration.json', 'r') as f:
        camera_params = json.load(f)
        camera_matrix = np.eye(3)
        camera_matrix[0, 0] = camera_params['intrinsics']['fx']
        camera_matrix[1, 1] = camera_params['intrinsics']['fy']
        camera_matrix[0, 2] = camera_params['intrinsics']['cx']
        camera_matrix[1, 2] = camera_params['intrinsics']['cy']
    # Load distortion coefficients
    with open(cam_param_dir / 'distortion_coeff.json', 'r') as f:
        dist_coeffs = json.load(f)
        dist_coeffs = np.array([dist_coeffs['k1'], dist_coeffs['k2'], dist_coeffs['p1'], dist_coeffs['p2'], dist_coeffs['k3']])
    return camera_matrix, dist_coeffs

def process_images(raw_img_dir, undistort_dir, img_dir, camera_matrix, dist_coeffs, scale_factor, downsample):
    count = 0
    # Clear the undistort directory
    if os.path.exists(undistort_dir):
        shutil.rmtree(undistort_dir)
    os.makedirs(undistort_dir, exist_ok=True)
    # Load raw images
    raw_images = natsorted([raw_img_dir / img for img in os.listdir(raw_img_dir)])
    
    image = cv2.imread(str(raw_images[0]))
    w, h = image.shape[1], image.shape[0]
    
    # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0.5, (w, h))
    
    for image_path in tqdm(raw_images):
        image = cv2.imread(str(image_path))
        # denoise the image
        image = cv2.bilateralFilter(image, 9, 75, 75)
        # image = cv2.GaussianBlur(image, (5, 5), 0)
        # image = cv2.medianBlur(image, 5)
        # image = cv2.undistort(image, camera_matrix, dist_coeffs)
        
        if downsample:
            new_width = int(image.shape[1] / scale_factor)
            new_height = int(image.shape[0] / scale_factor)
            # Resize the image
            # downsampled_image = cv2.resize(ud_image, (new_width, new_height))
            # # Crop the image
            cropped_image = crop_images(image, x, y, width, height)
            # newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 0, (w, h))
            # _, ud_image = undistort_image(image, newCameraMatrix, camera_matrix, dist_coeffs, roi)
            ud_image = cv2.undistort(cropped_image, camera_matrix, dist_coeffs)
            # cv2.imwrite(str(img_dir / image_path.name), downsampled_image)
            cv2.imwrite(str(img_dir / f'{count}.jpg'), ud_image)
            count += 1    
        # cv2.imwrite(str(undistort_dir / f'{count}.jpg'), ud_image)
    # return newCameraMatrix
    
def undistort_image(image, newCameraMatrix, camera_matrix, dist_coeffs, roi):
    
    ud_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, newCameraMatrix)
    x, y, w, h = roi
    cropped_ud_image = ud_image[y:y+h, x:x+w]
    return cropped_ud_image, ud_image

def crop_images(image, x, y, width, height):
    cropped_image = image[y:y+height, x:x+width]
    return cropped_image   
    
def load_poses(raw_poses_dir):
    load_ndi_poses(raw_poses_dir)
    load_phantom_poses(raw_poses_dir)
    load_zed_poses(raw_poses_dir)


def load_ndi_poses(poses_dir):
    # Load NDI poses write into poses_gt.txt
    with open(poses_dir / 'NDIHand.json', 'r') as f:
        ndi_poses = json.load(f)
    ndi_poses = dict(natsorted(ndi_poses.items()))
    count = 0
    with open(root_dir / 'poses_gt.txt', 'w') as f, open(root_dir / 'timestamp.txt', 'w') as t:
        for pose in ndi_poses.items():
            t.write(str(pose[0]) + '\n')
            tmp = np.array(pose[1])
            trans_matrix = tmp.reshape((4, 4))
            if np.isnan(trans_matrix).any():
                continue
            quat = R.from_matrix(trans_matrix[:3, :3]).as_quat()        
            t_vec = trans_matrix[:3, 3] / 1000
            f.write(f'{count} {t_vec[0]} {t_vec[1]} {t_vec[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n')
            count += 1


def load_phantom_poses(poses_dir):
    # Load Phantom poses write into phantom_poses_gt.txt
    poses_dir = root_dir / 'raw_poses'
    with open(poses_dir / 'NDISkull.json', 'r') as f:
        ndi_poses = json.load(f)
    # ndi_poses = natsorted(ndi_poses, key=lambda x: x)
    ndi_poses = dict(natsorted(ndi_poses.items()))
    # print(ndi_poses)
    count = 0
    with open(root_dir / 'phantom_poses_gt.txt', 'w') as f, open(root_dir / 'timestamp.txt', 'w') as t:
        for pose in ndi_poses.items():
            t.write(str(pose[0]) + '\n')
            
            tmp = np.array(pose[1])
            trans_matrix = tmp.reshape((4, 4))
            if np.isnan(trans_matrix).any():
                continue
            quat = R.from_matrix(trans_matrix[:3, :3]).as_quat()        
            t_vec = trans_matrix[:3, 3] / 1000
            f.write(f'{count} {t_vec[0]} {t_vec[1]} {t_vec[2]} {quat[0]} {quat[1]} {quat[2]} {quat[3]}\n')
            count += 1


def load_zed_poses(poses_dir):
    # Load ZED poses write into zed_poses_gt.txt
    poses_dir = root_dir / 'raw_poses'
    with open(poses_dir / 'zed.json', 'r') as f:
        zed_poses = json.load(f)

    # ndi_poses = natsorted(ndi_poses, key=lambda x: x)
    zed_poses = dict(natsorted(zed_poses.items()))
    # print(ndi_poses)
    count = 0
    with open(root_dir / 'zed_poses_gt.txt', 'w') as f:
        for pose in zed_poses.items():
            tmp = np.array(pose[1])
            if np.isnan(tmp).any():
                continue
            f.write(f'{count} {tmp[0]} {tmp[1]} {tmp[2]} {tmp[3]} {tmp[4]} {tmp[5]} {tmp[6]}\n')
            count += 1


def adjust_camera_params(new_param_dir, param_dir, scale_factor):
    new_camera_params = {"intrinsics": {"fx":0, "fy":0, "cx":0, "cy":0}, "fps": 25}
    with open(new_param_dir / 'calibration.json', 'w') as f, open(param_dir / 'calibration.json', 'r') as f_origin:
        camera_params = json.load(f_origin)
        new_camera_params['intrinsics']['fx'] = camera_params['intrinsics']['fx']/scale_factor
        new_camera_params['intrinsics']['fy'] = camera_params['intrinsics']['fy']/scale_factor
        new_camera_params['intrinsics']['cx'] = camera_params['intrinsics']['cx']/scale_factor - x
        new_camera_params['intrinsics']['cy'] = camera_params['intrinsics']['cy']/scale_factor - y
        json.dump(new_camera_params, f, indent=4)
        
    with open(param_dir / 'distortion_coeff.json', 'r') as f_origin, open(new_param_dir / 'distortion_coeff.json', 'w') as f:
        dist_coeffs = json.load(f_origin)
        json.dump(dist_coeffs, f, indent=4)
    
# def adjust_camera_params(newCameraMatrix, scale_factor):
#     new_camera_params = {"intrinsics": {"fx":0, "fy":0, "cx":0, "cy":0}, "fps": 25}
#     with open(modified_cam_param_dir / 'calibration.json', 'w') as f:
#         new_camera_params['intrinsics']['fx'] = newCameraMatrix[0, 0]/scale_factor
#         new_camera_params['intrinsics']['fy'] = newCameraMatrix[1, 1]/scale_factor
#         new_camera_params['intrinsics']['cx'] = newCameraMatrix[0, 2]/scale_factor  - x
#         new_camera_params['intrinsics']['cy'] = newCameraMatrix[1, 2]/scale_factor  - y
#         json.dump(new_camera_params, f, indent=4)
       
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dataset Processing')
    parser.add_argument('--root_dir', type=str, default='./data/phantom_data_521/left_CommonMotion', help='Root directory of the dataset')
    parser.add_argument('--calib_dir', type=str, default='./data/phantom_data_521/', help='Root directory of the calibration data')
    parser.add_argument('--downsample', action='store_true', help='Downsample the images to half size')
    args = parser.parse_args()
    
    root_dir = Path(args.root_dir)
    calib_root = Path(args.calib_dir)
    raw_img_dir = root_dir / 'raw_images'
    undistort_dir = root_dir / 'undistort_images'
    cam_param_dir = calib_root / 'cam_params'
    modified_cam_param_dir = root_dir / 'cam_params'
    raw_poses_dir = root_dir / 'raw_poses'
    img_dir = root_dir / 'images'
    print(root_dir)
    # os.makedirs(raw_img_dir, exist_ok=True)
    # os.makedirs(cam_param_dir, exist_ok=True)
    # os.makedirs(raw_poses_dir, exist_ok=True)
    os.makedirs(modified_cam_param_dir, exist_ok=True)
    os.makedirs(undistort_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    
    # Crop parameters
    x=345
    y=185
    width=620 
    height=620
    scale_factor = 1
    camera_matrix, dist_coeffs = load_cam_params(cam_param_dir)
    
    if args.downsample:
        adjust_camera_params(modified_cam_param_dir, cam_param_dir, scale_factor)
        camera_matrix, dist_coeffs = load_cam_params(modified_cam_param_dir)
        
    process_images(raw_img_dir, undistort_dir, img_dir, camera_matrix, dist_coeffs, scale_factor, downsample=args.downsample)
    
    # if args.downsample:
        # adjust_camera_params(modified_cam_param_dir, cam_param_dir, scale_factor)
        # adjust_camera_params(newCameraMatrix, scale_factor)
    load_poses(raw_poses_dir)
    print('Done!')