import sys
import numpy as np
import copy
import copy
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import os
import seaborn as sns
import argparse

def visulizePoints(point_array):
    x = point_array[:, 0]
    y = point_array[:, 1]
    z = point_array[:, 2]

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # source_temp.paint_uniform_color([1, 0.706, 0])
    # target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    # print("")
    # print(
    #     "1) Please pick at least three correspondences using [shift + left click]"
    # )
    # print("   Press [shift + right click] to undo point picking")
    # print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def manual_registration(source, target):

    print("Visualization of two point clouds before manual alignment")
    # draw_registration_result(source, target, np.identity(4))
    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
    assert (len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source), 2))
    corr[:, 0] = picked_id_source
    corr[:, 1] = picked_id_target

    for idx in picked_id_source:
        point = source.points[idx]
        # print(np.linalg.norm(point))
        print(point)


    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.registration.TransformationEstimationPointToPoint(with_scaling=True)
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    # reg_p2p = o3d.registration.registration_icp(
    #     source, target, threshold, trans_init,
    #     o3d.registration.TransformationEstimationPointToPoint(with_scaling=True))
    draw_registration_result(source, target, trans_init)
    save_path = args.save + '/gs_surfel/trans_init.npy'
    np.save(save_path, trans_init)
    print("Transformation matrix saved as 'trans_init.npy'")
    source.transform(trans_init)
    findCorrespondences(source, target)
    return trans_init

def PointsRegistration(source_points, target_points):
    # Create Open3D point cloud objects
    source_pcd = o3d.geometry.PointCloud()
    target_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    res = manual_registration(source_pcd, target_pcd)
    # Get the aligned source point cloud
    aligned_source_pcd = source_pcd.transform(res.transformation)

    # Print the transformation matrix
    print("Transformation matrix:")
    print(res.transformation)

    # Visualize the point clouds
    o3d.visualization.draw_geometries([target_pcd, aligned_source_pcd])
    print(res)
    s_p = np.asarray(aligned_source_pcd.points)
    t_p = np.asarray(target_pcd.points)
    distances = np.linalg.norm(s_p - t_p, axis=1)
    mean_distance = np.mean(distances)
    print('mean_distance: ', mean_distance)

def AutoRegistration(source_points, target_points):
    pcd1_down = source_points.voxel_down_sample(voxel_size=0.05)
    pcd2_down = target_points.voxel_down_sample(voxel_size=0.05)
    o3d.visualization.draw_geometries([pcd1_down])
    max_correspondence_distance = 10

    reg_p2p = o3d.registration.registration_icp(
    pcd1_down, pcd2_down, 
    max_correspondence_distance, np.identity(4),
    o3d.registration.TransformationEstimationPointToPoint(with_scaling=True),
    o3d.registration.ICPConvergenceCriteria(max_iteration=2000)
    )   
    print(reg_p2p.transformation)
    R = reg_p2p.transformation[:3, :3]
    scale = np.linalg.norm(R, axis=0)
    print(scale)
    aligned_source_pcd = source_points.transform(reg_p2p.transformation)
    o3d.visualization.draw_geometries([target_points, aligned_source_pcd])

    # findCorrespondences(aligned_source_pcd, target_points)
    
def findCorrespondences(transformed_source_pcd, target_pcd):
    # Get points from both point clouds
    points1 = np.asarray(transformed_source_pcd.points)
    points2 = np.asarray(target_pcd.points)

    # Find nearest neighbors in pcd2 for each point in pcd1_transformed
    search_tree = o3d.geometry.KDTreeFlann(target_pcd)
    correspondences = []
    distances = []
    for i in range(len(points1)):
        [k, idx, _] = search_tree.search_knn_vector_3d(points1[i], 1)
        correspondences.append([i, idx[0]])
        distances.append(np.linalg.norm(points1[i] - points2[idx[0]]))
        
    # # Print the correspondences
    # for corr in correspondences:t
    #     print("Point in pcd1: {}, Point in pcd2: {}".format(corr[0], corr[1]))
        
    print("Mean distance between corresponding points: ", np.mean(distances))
    # hausdorff_distance = calculate_hausdorff_distance(transformed_source_pcd, target_pcd)
    print("Hausdorff distance: ", np.max(distances))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--source', type=str, required=True, help='Path to the source point cloud file')
    parser.add_argument('--target', type=str, required=True, help='Path to the target point cloud file')
    parser.add_argument('--save', type=str, required=True, help='Path to save registration result')

    args = parser.parse_args()

    # data_path = '../data/reconstruction_eval/'
    pcd_target = o3d.io.read_point_cloud(args.target)
    pcd_target.points = o3d.utility.Vector3dVector(np.asarray(pcd_target.points) / 1000)
    
    pcd_source = o3d.io.read_point_cloud(args.source)
    # Filter outliers of pcd_source using K-nearest neighbors
    k = 5  # Number of neighbors to consider
    pcd_source_filtered, _ = pcd_source.remove_radius_outlier(nb_points=k, radius=0.1)
    # Clip the pcd source by depth
    depth_min = 0.  # Minimum depth value
    depth_max = 1.8  # Maximum depth value
    pcd_source_clipped = pcd_source_filtered.crop(o3d.geometry.AxisAlignedBoundingBox([float('-inf'), float('-inf'), depth_min], [float('inf'), float('inf'), depth_max]))
    
    # Perform registration with the clipped point cloud
    reg_p2p = manual_registration(pcd_source_clipped, pcd_target)

    # reg_p2p = manual_registration(pcd_source, pcd_target)
    # print(reg_p2p)
    # AutoRegistration(pcd_source, pcd_target)
    
    # reg_res = reg_p2p.transformation
    # np.save('reg_res.npy', reg_p2p.transformation)
    # reg_res = np.load(data_path + 'reg_res.npy')
    # inv_reg = sol.invTransformation(reg_res)
    # pcd_target.transform(inv_reg)
    # picked_id_target = pick_points(pcd_target)
    # target_pickpoints = []
    # for idx in picked_id_target:
    #     point = pcd_target.points[idx]
    #     target_pickpoints.append(point)
    # target_pickpoints = np.vstack(target_pickpoints)


    # PointsRegistration(pcd_source, pcd_target)