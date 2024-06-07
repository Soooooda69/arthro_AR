import numpy as np
from collections import defaultdict
from spatialmath.base import *
from spatialmath import SO3, SE3, Quaternion

class Camera(object):
    def __init__(self, model, intrinsic, width, height):
        self.model = model
        if model == 'PINHOLE':
            self.fx = intrinsic[0]
            self.fy = intrinsic[1]
            self.cx = intrinsic[2]
            self.cy = intrinsic[3]
        elif model == 'SIMPLE_RADIAL':
            self.fx = intrinsic[0]
            self.fy = intrinsic[0]
            self.cx = intrinsic[1]
            self.cy = intrinsic[2]
            self.k1 = intrinsic[3]
            self.k2 = intrinsic[3]
            self.distortion = np.array([self.k1, self.k2 ,0 ,0 ,0])
            
        self.intrinsic_mtx = np.array([
            [self.fx, 0, self.cx], 
            [0, self.fy, self.cy], 
            [0, 0, 1]])
        
        self.frustum_near = 0.1
        self.frustum_far = 2

        self.width = width
        self.height = height

class Frame(object):
    def __init__(self, idx, pose, mask, image, cam, timestamp, 
            pose_covariance=np.identity(6)):
        self.idx = idx
        # g2o.Isometry3d
        if pose.isSE:
            self.pose = pose
        else:
            print('Pose is not SE3!')
        self.rotation = self.pose.R
        self.position = self.pose.t
        self.feature = ImageFeature(image, mask, idx)
        self.cam = cam
        self.intrinsic = [self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy] # [fx, fy, cx, cy]
        self.hfov = 2 * np.arctan(self.cam.width / (2 * self.cam.fx))
        self.intrinsic_mtx = self.cam.intrinsic_mtx
        self.timestamp = timestamp
        
        self.pose_covariance = pose_covariance

        self.transform_matrix = self.pose.inv() # transform from world frame to camera frame
        self.itransform_matrix = self.pose # transform from camera frame to world frame
        self.projection_matrix = (
            self.cam.intrinsic_mtx*self.transform_matrix.t)  # from world frame to image
    
    def __hash__(self):
        return self.idx

    def transform(self, points):    # from world coordinates
        '''
        Transform points from world coordinates frame to camera frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        return self.transform_matrix * points
    
    def itransform(self, points):   # from camera coordinates
        '''
        Transform points from camera coordinates frame to world frame.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        '''
        return self.itransform_matrix * points
    
    def project(self, points): 
        '''
        Project points from camera frame to image's pixel coordinates.
        Args:
            points: a point or an array of points, of shape (3,) or (3, N).
        Returns:
            Projected pixel coordinates, and respective depth.
        '''
        projection = self.cam.intrinsic_mtx.dot(points / points[-1:])
        return projection[:2], points[-1]

    def unproject(self, points, depth_image):
        '''
        Unproject points from image's pixel coordinates to camera frame.
        Args:
            points: a point or an array of points, of shape (,2) or (N, 2).
            depth: a scalar or an array of scalars, of shape (1,) or (1, N).
        Returns:
            Unprojected points in camera frame. (N, 3)
        '''
        x_d, y_d = points[:, 0], points[:, 1]
        fx, fy, cx, cy = self.cam.fx, self.cam.fy, self.cam.cx, self.cam.cy

        # depths = depth_image[y_d.astype(int), x_d.astype(int)]
        depths = depth_image[x_d.astype(int), y_d.astype(int)]
        x = ((x_d - cx) * depths / fx)[:, None]
        y = ((y_d - cy) * depths / fy)[:, None]
        z = depths[:, None]

        points_3d = np.hstack([y, x, z]).transpose()
        return self.itransform(points_3d).transpose()
    
    def get_keypoint(self, i):
        return self.feature.get_keypoint(i)
    def get_descriptor(self, i):
        return self.feature.get_descriptor(i)
    def get_color(self, pt):
        return self.feature.get_color(pt)

class ImageFeature(object):
    def __init__(self, image, mask, idx):
        self.image = image
        self.mask  = mask
        self.idx = idx
        self.height, self.width = image.shape[:2]
        
        self.keypoints_info = defaultdict() #{keypoints_id: (keypoints, descriptors)}
        self.keypoints_ids = []
        self.keypoints = [] # list of keypoints 2d coordinates
        self.descriptors = []

    def update_keypoints_info(self):
        self.keypoints_ids = [keypoint_id for keypoint_id in self.keypoints_info.keys()]
        self.keypoints = [self.keypoints_info[keypoint_id][0] for keypoint_id in self.keypoints_ids]
        self.descriptors = [self.keypoints_info[keypoint_id][1] for keypoint_id in self.keypoints_ids]
        