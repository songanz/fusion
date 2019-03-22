# camera-ready

import os
import numpy as np

import sys
root_dir = "/home/songanz/Documents/Git_repo/fusion/"  # change this for your own usage
sys.path.append(root_dir + "Open3D/build/lib")  # NOTE! you'll have to adapt this for your file structure
from open3d import *


def draw_geometries_dark_background(geometries):
    vis = Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

project_dir = root_dir # NOTE! you'll have to adapt this for your file structure
data_dir = project_dir + "data/kitti/object/training/"
img_dir = data_dir + "image_2/"
lidar_dir = data_dir + "velodyne/"

img_paths = []
label_paths = []
lidar_paths = []
img_names = os.listdir(img_dir)
for step, img_name in enumerate(img_names):
    img_id = img_name.split(".png")[0]

    img_path = img_dir + img_name
    img_paths.append(img_path)

    lidar_path = lidar_dir + img_id + ".bin"
    lidar_paths.append(lidar_path)

for lidar_path in lidar_paths:
    point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

    print (lidar_path)
    print (point_cloud.shape)

    point_cloud_xyz = point_cloud[:, 0:3]

    pcd = PointCloud()
    pcd.points = Vector3dVector(point_cloud_xyz)
    pcd.paint_uniform_color([0.65, 0.65, 0.65])

    draw_geometries_dark_background([pcd])
