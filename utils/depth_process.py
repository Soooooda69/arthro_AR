import statistics
import numpy as np
from datetime import datetime
from pathlib import Path
import cv2
import os
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from components import Camera, Frame
from spatialmath.base import *
from spatialmath import SE3
import sys
from sklearn.linear_model import RANSACRegressor
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../models')))
from depth_anything.depth_anything.dpt import DepthAnything
from depth_anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet 
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

class DepthEstimatorDepthAnything():
    def __init__(self, encoder:['vits', 'vitb', 'vitl'], multiplier=1) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(self.DEVICE).eval()
        self.mapPoint_xyzs = {}
        self.current_frame = None
        # self.mapPoint_xys = []
        self.transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
        ])
        
        self.disp = None
        self.cv_image = None
        self.save_id = 0
    
    def __call__(self, image):
        # image = image.permute(1, 2, 0).detach().cpu().numpy()*255
        # cv_image = image.astype(np.uint8)
        self.cv_image = image
        h, w = image.shape[:2]
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
        image = self.transform({'image': image})['image']
        image = torch.from_numpy(image).unsqueeze(0).to(self.DEVICE)
        with torch.no_grad():
            disp = self.depth_anything(image)
        disp = F.interpolate(disp[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
        self.disp = disp.detach().cpu().numpy()
        # self.disp = (self.disp - self.disp.min()) / (self.disp.max() - self.disp.min()) * 1
        # self.disp = np.clip(self.disp, np.mean(self.disp)- 2*np.std(self.disp), None)
        # self.disp = (self.disp - self.disp.min()) / (self.disp.max() - self.disp.min())
        
    def visualize(self, path):
        disp_cv = (self.disp - self.disp.min()) / (self.disp.max() - self.disp.min()) * 255.0
        disp_cv = disp_cv.astype(np.uint8)
        depth_color = cv2.applyColorMap(disp_cv, cv2.COLORMAP_INFERNO)
        combined_results = cv2.hconcat([self.cv_image, depth_color])
        
        cv2.imwrite(path, combined_results)
        self.save_id += 1

    def save_disp(self, path):
        np.save(path, self.disp)
        
class DepthProcess:
    def __init__(self, name='', output_folder='./output') -> None:
        # Other Settings
        now = datetime.now()
        self.exp_name = name
        self.exp_root = Path(output_folder) / self.exp_name
        self.exp_colmapStyle_root = Path(output_folder) / self.exp_name / 'colmap_data'
        self.mask = None
        self.cams = {}
        self.frames = {}
        self.disp = {}
        self.xyzs = {}
        self.rgbs = {}
        self.co_frames = {}
        self.prev_res = None
        
    def load_data(self):
        def load_frames(sparse_root, image_root, disp_root):
            '''
            # Camera list with one line of data per camera:
            #   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
            # Number of cameras: 3
            '''
            with open(sparse_root / 'cameras.txt', 'r') as f_cam:
                for camera_info in f_cam.readlines():
                    if camera_info.startswith('#'):
                        continue
                    camera_info = camera_info.strip().split(' ')
                    camera_id = int(camera_info[0])
                    model = camera_info[1]
                    width = int(camera_info[2])
                    height = int(camera_info[3])
                    intrinsic = [float(param) for param in camera_info[4:]]
                    # Create a Camera object and add it to the dataset
                    cam = Camera(model, intrinsic, width, height)
                    self.cams[camera_id] = cam
                    
            with open(sparse_root / 'images.txt', 'r') as f_img:
                while True:
                    line = f_img.readline()
                    if not line:
                        break
                    if line.startswith('#'):
                        continue
                    pose_info = line.strip().split(' ')
                    # image_id = int(pose_info[0])
                    camera_id = int(pose_info[8])
                    image_id = int(pose_info[9].split('.')[0].split('/')[-1])
                    qw, qx, qy, qz, tx, ty, tz = map(float, pose_info[1:-2])
                    elems = f_img.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                        tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    pose = SE3.Rt(q2r([qw,qx,qy,qz]), np.array([tx, ty, tz]))
                    # colmap pose is from camera to world, we need to invert it
                    pose = pose.inv()
                    image = cv2.imread(str(image_root / f"{image_id}.png"))
                    mask = None
                    timestamp = None
                    # descs = load_descs(desc_root, image_id)
                    cam = self.cams[camera_id]
                    
                    frame = Frame(image_id, pose, mask, image, cam, timestamp)
                    frame.feature.keypoints_info = {point3D_ids[i]: (xys[i], 0) for i in range(len(xys)) if point3D_ids[i] != -1}
                    frame.feature.update_keypoints_info()
                    disp = load_disp(disp_root, image_id)
                    self.frames[image_id] = frame
                    self.disp[image_id] = disp
                    
        def load_points3D(sparse_root):

            with open(sparse_root / 'points3D.txt', "r") as fid:
                while True:
                    line = fid.readline()
                    if not line:
                        break
                    line = line.strip()
                    if len(line) > 0 and line[0] != "#":
                        elems = line.split()
                        mapPoint_id = int(elems[0])
                        xyz = np.array(tuple(map(float, elems[1:4])))
                        rgb = np.array(tuple(map(int, elems[4:7])))
                        frame_id = np.array(tuple(map(int, elems[8::2])))
                        if mapPoint_id != -1:
                            self.xyzs[mapPoint_id] = xyz
                            self.rgbs[mapPoint_id] = rgb
                            self.co_frames[mapPoint_id] = frame_id
                            
        def load_disp(disp_root, idx):
            return np.load(disp_root / f'{idx}.npy')
        
        # self.exp_colmapStyle_root.mkdir(parents=True, exist_ok=True)
        # desc_root = self.exp_colmapStyle_root / 'descriptors'
        load_frames(sparse_root, image_root, disp_root)
        load_points3D(sparse_root)
        # filter points by covisible frames
        less_visiable_frames = []
        for key, value in self.co_frames.items():
            if len(value) < len(self.frames)*0.1:
                less_visiable_frames.append(key)
                
        print('Total points:',len(self.xyzs))
        self.xyzs = {key: value for key, value in self.xyzs.items() if key not in less_visiable_frames}
        points = np.vstack([xyz for xyz in self.xyzs.values()])
        print('Number of points:', points.shape[0])
        # filtered_points = self.remove_outliers(points)
        # print('Number of points after removing outliers:', filtered_points.shape[0])
        self.xyzs = {key: value for key, value in self.xyzs.items() if value in points}

    # Remove outliers using k-nearest neighbors
    def remove_outliers(self, points, k=5):
        # Fit the k-nearest neighbors model
        nbrs = NearestNeighbors(n_neighbors=k)
        nbrs.fit(points)
        # Compute the distances to the k nearest neighbors
        distances, _ = nbrs.kneighbors(points)
        # Compute the mean distance to the k nearest neighbors
        mean_distances = np.mean(distances, axis=1)
        threshold = np.percentile(mean_distances, 80)
        # Find the indices of points with distances greater than the threshold
        outlier_indices = np.where(mean_distances > threshold)[0]
        # Remove the outliers from the points array
        filtered_points = np.delete(points, outlier_indices, axis=0)
        return filtered_points   
    
    def scale_recovery(self, xyzs, frame, disp):
        def cost(params):
            s,t = params
            return np.linalg.norm(s * (1/depth_anything_output)+t - mapPoint_depth)
        
        mapPoint_depth = []
        depth_anything_output = []
        # frame.feature.keypoints_info = {key: value for key, value in frame.feature.keypoints_info.items() if key % 10 == 0}
        for keypoints_id, (keypoints, _) in frame.feature.keypoints_info.items():
            
            x, y  = keypoints[0], keypoints[1]
            if x < 0 or x >= disp.shape[1] or y < 0 or y >= disp.shape[0]:
                continue
            if self.mask:
                if cam_mask[y.astype(int), x.astype(int)][0] == 0:
                    print('Masked point')
                    continue
            if keypoints_id not in self.xyzs.keys():
                continue
            
            disp_value = disp[y.astype(int), x.astype(int)]
            if disp_value < np.median(disp):
                continue
            
            depth_anything_output.append(disp_value)
            point3d = frame.transform(xyzs[keypoints_id][:,None])
            mapPoint_depth.append(point3d[2])
        depth_anything_output = np.array(depth_anything_output)
        mapPoint_depth = np.array(mapPoint_depth).flatten()
        # # Remove values larger than the median
        # median_depth = np.median(1/depth_anything_output)
        # mask = (1 / depth_anything_output) <= median_depth
        # depth_anything_output = depth_anything_output[mask]
        # mapPoint_depth = mapPoint_depth[mask]
        
        # print(depth_anything_output)
        # print(mapPoint_depth)
        # s0 = statistics.median(mapPoint_depth) / (1/statistics.median(disp.flatten()))
        s0 = 1
        t0 = 0
        params = [s0, t0]
        res = minimize(cost, params, method='Nelder-Mead', options={'disp': True, 'xatol': 1e-5, 'fatol': 1e-4})

        print('Scale:', res.x[0], 'Translation:', res.x[1], 'image:', frame.idx)
        if res.x[0] < 0:
            print('Negative scale')
            breakpoint()
            res = self.prev_res
            
            
        depth = res.x[0] * (1/disp) + res.x[1]
        self.prev_res = res
        # depth = np.clip(depth, None, 3 * np.median(depth))
        
        # # Plot the distribution of depth_anything_output
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        # ax1.hist(depth, bins=50)
        # ax1.set_xlabel('Depth')
        # ax1.set_ylabel('Frequency')
        # ax1.set_title('Distribution of scaled_depth')

        # ax2.hist(mapPoint_depth, bins=50)
        # ax2.set_xlabel('Depth')
        # ax2.set_ylabel('Frequency')
        # ax2.set_title('Distribution of mapPoint_depth')
        # plt.savefig('depth_dist.png')

        return depth
    
                
    def save_projection(self, frame, depth, ply_root):
        # test & visualize depth
        # frame = slam_structure.all_frames[section_indices[start_frame]]
        image = frame.feature.image
        new_points_2d = []
        color = []
        for x in range(depth.shape[0]):
            for y in range(depth.shape[1]):
                if self.mask:
                    if cam_mask[x, y, 0] == 0:
                        continue
                new_points_2d.append([x, y])
                color.append(image[x, y])

        new_points_2d = np.vstack(new_points_2d)
        color = np.vstack(color).astype(np.uint8)
        new_points_3d = frame.unproject(new_points_2d, depth)
        
        point_cloud = pd.DataFrame({
            'x': new_points_3d[:, 0],
            'y': new_points_3d[:, 1],
            'z': new_points_3d[:, 2],
            'red': color[:, 2],
            'green': color[:, 1],
            'blue': color[:, 0]
        })
        # Delete points where color is black
        if self.mask:
            point_cloud = point_cloud[point_cloud['red'] != 0]
            point_cloud = point_cloud[point_cloud['green'] != 0]
            point_cloud = point_cloud[point_cloud['blue'] != 0]
        
        pynt_cloud = PyntCloud(point_cloud)
        pynt_cloud.to_file(str(ply_root) + f'/{frame.idx}.ply')  

def disp_inference(image_root):
    
    depth_estimator = DepthEstimatorDepthAnything(encoder='vitl')
    for image in tqdm(os.listdir(image_root)):
        img = cv2.imread(str(image_root / image))
        depth_estimator(img)
        depth_estimator.visualize(str(disp_root / f'{image}') )
        idx = image.split('.')[0]
        depth_estimator.save_disp(disp_root / f'{idx}.npy')

def main(dep, mask):
    
    # Load the data
    dep.load_data()
    position = np.vstack([xyz for xyz in dep.xyzs.values()])
    color = np.vstack([rgb for rgb in dep.rgbs.values()]).astype(np.uint8)
    for idx in sorted(dep.frames.keys()):
        frame = dep.frames[idx]
        disp = dep.disp[idx]
        if mask:
            disp[cam_mask[:,:,0] == 0] = 1
        depth = dep.scale_recovery(dep.xyzs, frame, disp)
        if depth is None:
            continue
        np.save(depth_root / f'{idx}.npy', depth)
        depth_cv = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_cv = depth_cv.astype(np.uint8)
        depth_color = cv2.applyColorMap((depth_cv).astype(np.uint8), cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(depth_video_root / f'depth_{idx}.png'), depth_color)
        
        dep.save_projection(frame, depth, ply_root)
        # breakpoint()
    # a,b = dep.global_scale_recovery()
    # for idx in sorted(dep.frames.keys()):
    #     frame = dep.frames[idx]
    #     disp = dep.disp[idx]
    #     depth = a * (1/disp) + b
    
    #     dep.save_projection(frame, depth, ply_root)
        # breakpoint()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Depth Process')
    parser.add_argument('--name', type=str, help='Name of the experiment')
    parser.add_argument('--output_folder', type=str, default='./output', help='Output folder path')
    parser.add_argument('--mask', action='store_true', help='Enable mask')
    
    args = parser.parse_args()
    
    # Create an instance of DepthProcess
    dep = DepthProcess(name=args.name, output_folder=args.output_folder)
    dep.mask = args.mask
    
    disp_root = dep.exp_root / 'disp'
    depth_root = dep.exp_root / 'depth'
    ply_root = dep.exp_root / 'ply'
    sparse_root = dep.exp_colmapStyle_root / 'sparse/0'
    image_root = dep.exp_colmapStyle_root / 'images'
    # undistort_image_root = dep.exp_colmapStyle_root / 'ud_images'
    depth_video_root = dep.exp_root / 'depth_video'
    os.makedirs(disp_root, exist_ok=True)
    os.makedirs(depth_root, exist_ok=True)
    os.makedirs(ply_root, exist_ok=True)
    os.makedirs(depth_video_root, exist_ok=True)
    # os.makedirs(undistort_image_root, exist_ok=True)
   
    disp_inference(image_root)
    
    main(dep, mask=args.mask)