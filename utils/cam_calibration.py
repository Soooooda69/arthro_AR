import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from natsort import natsorted
import argparse

import concurrent.futures

def corner_detection(calibration_dir, board):
    calibration_images = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.endswith('.jpg')]
    image = cv2.imread(calibration_images[0])
    image_size = (image.shape[1], image.shape[0])
    all_corners = []
    all_ids = []

    def process_image(image_path):
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)
        if len(corners) > 6:
            # Refine detected markers
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image=gray,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=None,
                cameraMatrix=None,
                distCoeffs=None
            )

            if len(corners) > 6:
                # Interpolate charuco corners
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board
                )

                if charuco_ids is not None and len(charuco_ids) > 6:
                    return charuco_corners, charuco_ids

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = tqdm(executor.map(process_image, calibration_images))

        for result in results:
            if result is not None:
                charuco_corners, charuco_ids = result
                all_corners.append(charuco_corners)
                all_ids.append(charuco_ids)

    if len(all_corners) < 10:
        print("Not enough valid frames for calibration.")
        return
    else:
        print(f"Found valid Charuco corners in {len(all_corners)} frames.")
    return all_corners, all_ids, image_size

# Function to generate object points for detected corners
def generate_charuco_object_points(charuco_board, charuco_ids):
    all_object_points = []
    for ids in charuco_ids:
        obj_points = charuco_board.chessboardCorners[ids.flatten(), :]
        all_object_points.append(obj_points)
    return all_object_points


def calibrate_camera_ransac(all_corners, all_ids, charuco_board, image_size, num_imgs, num_iterations=100):
    best_camera_matrix = None
    best_dist_coeffs = None
    best_error = float('inf')
    best_inliers = []
    # Generate object points
    all_object_points = generate_charuco_object_points(charuco_board, all_ids)
    for _ in tqdm(range(num_iterations)):
        sample_indices = natsorted(random.sample(range(len(all_corners)), min(num_imgs, len(all_corners))))
        sample_corners = [all_corners[i] for i in sample_indices]
        sample_ids = [all_ids[i] for i in sample_indices]
        sample_object_points = [all_object_points[i] for i in sample_indices]
        retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            sample_corners, sample_ids, charuco_board, image_size, None, None
        )

        if retval:
            total_error = 0
            total_points = 0

            for i in range(len(sample_corners)):
                if sample_corners[i] is not None and len(sample_corners[i]) > 0:                    
                    charuco_corners_3d = sample_object_points[i]
                    imgpoints2, _ = cv2.projectPoints(charuco_corners_3d, rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
                    error = cv2.norm(sample_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                    total_error += error
                    total_points += 1

            avg_error = total_error / total_points

            if avg_error < best_error:
                best_error = avg_error
                best_camera_matrix = camera_matrix
                best_dist_coeffs = dist_coeffs
                best_inliers = sample_indices

    return best_camera_matrix, best_dist_coeffs, best_inliers, best_error

def eval_by_projection(calibration_dir, board, camera_matrix, dist_coeffs, video_path):
    calibration_images = [os.path.join(calibration_dir, f) for f in os.listdir(calibration_dir) if f.endswith('.jpg')]
    # frame_interval = 5  # Specify the frame interval for subsampling
    # calibration_images = calibration_images[::frame_interval]

    # Choose a random sample of images
    num_images_to_choose = 50  # Specify the number of images to choose
    calibration_images = random.sample(calibration_images, num_images_to_choose)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    image = cv2.imread(calibration_images[0])
    image_size = (image.shape[1], image.shape[0])
    video_writer = cv2.VideoWriter(video_path, fourcc, 5, image_size)

    for image_path in tqdm(calibration_images):
        image = cv2.imread(image_path)
        # image = cv2.undistort(image, camera_matrix, dist_coeffs)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, ARUCO_DICT)

        if len(corners) > 0:
            # Refine detected markers
            corners, ids, _, _ = cv2.aruco.refineDetectedMarkers(
                image=gray,
                board=board,
                detectedCorners=corners,
                detectedIds=ids,
                rejectedCorners=None,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )

            if len(corners) > 0:
                # Interpolate charuco corners
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    markerCorners=corners,
                    markerIds=ids,
                    image=gray,
                    board=board
                )

                if ret > 3:
                    # Draw detected markers
                    image_markers = cv2.aruco.drawDetectedMarkers(image.copy(), corners, ids)

                    # Estimate pose and reproject markers
                    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charucoCorners=charuco_corners,
                        charucoIds=charuco_ids,
                        board=board,
                        cameraMatrix=camera_matrix,
                        distCoeffs=dist_coeffs,
                        rvec=None,
                        tvec=None
                    )

                    if ret:
                        # Draw reprojected markers
                        image_reproj = cv2.aruco.drawDetectedCornersCharuco(
                            image=image_markers,
                            charucoCorners=charuco_corners,
                            charucoIds=charuco_ids,
                            cornerColor=(0, 255, 0)  # Green color for reprojected markers
                        )
                        # Draw the axis for the origin
                        cv2.aruco.drawAxis(
                            image=image_reproj,
                            cameraMatrix=camera_matrix,
                            distCoeffs=dist_coeffs,
                            rvec=rvec,
                            tvec=tvec,
                            length=SQUARE_LENGTH * 2
                        )
                        # Write image_reproj to video
                        video_writer.write(image_reproj)
    video_writer.release()
                        # Display the image with detected and reprojected markers
                        # plt.imshow(cv2.cvtColor(image_reproj, cv2.COLOR_BGR2RGB))
                        # plt.axis('off')
                        # plt.show()
                        # cv2.imwrite(os.path.join(corners_image_path, f'{os.path.basename(image_path)}'), image_reproj)

def save_result(save_dir, camera_matrix, dist_coeffs):
    # Save the calibration results
    # np.savez('calibration.npz', camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)# Convert camera_matrix to a dictionary
    calibration_dict = { 
        "intrinsics": {
            "fx": camera_matrix[0,0],
            "fy": camera_matrix[1,1],
            "cx": camera_matrix[0,2],
            "cy": camera_matrix[1,2]
        },
        "fps" : 25
    }

    dist_coeffs = dist_coeffs.flatten()
    distort_dict = {
        "k1": dist_coeffs[0],
        "k2": dist_coeffs[1],
        "p1": dist_coeffs[2],
        "p2": dist_coeffs[3],
        "k3": dist_coeffs[4]
    }
    # Save the camera_matrix as JSON
    with open(os.path.join(save_dir,'calibration.json'), 'w') as f:
        json.dump(calibration_dict, f)

    with open(os.path.join(save_dir,'distortion_coeff.json'), 'w') as f:
        json.dump(distort_dict, f)
    print("Done saving!")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Dataset Processing')
    parser.add_argument('--root_dir', type=str, default='./data/phantom_data_521', help='Root directory of the dataset')
    parser.add_argument('--calib_dir', type=str, default='./data/phantom_data_521/calib_begin', help='Root directory of the calibration data')
    args = parser.parse_args()
    # calibration_dir = os.path.join(args.root_dir, 'calib_end')
    save_dir = os.path.join(args.root_dir, 'cam_params')
    os.makedirs(save_dir, exist_ok=True)

    # ChArUco board parameters
    CHARUCOBOARD_ROWCOUNT = 8
    CHARUCOBOARD_COLCOUNT = 11
    ARUCO_DICT = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    SQUARE_LENGTH = 0.003  # Checker size in meters
    MARKER_LENGTH = 0.002 # Marker size in meters

    # Create the ChArUco board
    charucoboard = cv2.aruco.CharucoBoard_create(
        squaresX=CHARUCOBOARD_COLCOUNT,
        squaresY=CHARUCOBOARD_ROWCOUNT,
        squareLength=SQUARE_LENGTH,
        markerLength=MARKER_LENGTH,
        dictionary=ARUCO_DICT
    )
    
    # Detect ChArUco corners
    all_corners, all_ids, image_size = corner_detection(args.calib_dir, charucoboard)
    # Calibrate camera using RANSAC
    best_camera_matrix, best_dist_coeffs, best_inliers, best_error = calibrate_camera_ransac(
        all_corners, all_ids, charucoboard, image_size, num_imgs=20, num_iterations=300
    )

    if best_camera_matrix is not None:
        print("Calibration successful with RANSAC.")
        print(f"Number of inliers: {len(best_inliers)}.")
        print("Camera Matrix:\n", best_camera_matrix)
        print("Distortion Coefficients:\n", best_dist_coeffs)
        print("Best Error:", best_error)
    else:
        print("Calibration failed.")
        
    save_result(save_dir, best_camera_matrix, best_dist_coeffs)
    
    eval_by_projection(args.calib_dir, charucoboard, best_camera_matrix, best_dist_coeffs, video_path=os.path.join(args.root_dir, 'reprojection_video.mp4'))