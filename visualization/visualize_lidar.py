# camera-ready

import os
import numpy as np

import sys
# root_dir = "/home/songanz/Documents/Git_repo/fusion/"  # change this for your own usage
# sys.path.append(root_dir + "Open3D/build/lib")  # NOTE! you'll have to adapt this for your file structure
from open3d import *


def draw_geometries_dark_background(geometries):
    vis = visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.show_coordinate_frame = True  # x, y, z axis will be rendered as red, green, and blue
    # opt.background_color = np.asarray([0, 0, 0])
    opt.background_color = np.asarray([1, 1, 1])
    # opt.background_color = np.asarray([0.5, 0.5, 0.5])
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

# project_dir = root_dir # NOTE! you'll have to adapt this for your file structure
# data_dir = project_dir + "data/kitti/object/training/"
img_dir = "/media/songanz/New Volume/data/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync/image_00/data_rect/"
lidar_dir = "/media/songanz/New Volume/data/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/"

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


# geometry.OrientedBoundingBox inputs: center, rotation R and extent in x, y and z direction
bbox = geometry.OrientedBoundingBox(np.array([0,0,0]), np.eye(3), np.array([1,1,1]))
cylinder = geometry.TriangleMesh.create_cylinder(radius=0.3, height=4.0)

lidar_path = "/media/songanz/New Volume/data/KITTI-360/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/0000000010.bin"
point_cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

print(lidar_path)
print("pcd shape: {}".format(point_cloud.shape))

point_cloud_xyz = point_cloud[:, 0:3]
pcd = geometry.PointCloud()
pcd.points = utility.Vector3dVector(point_cloud_xyz)
pcd.paint_uniform_color([0.2, 0.2, 0.2])

polygon_cut = visualization.SelectionPolygonVolume()
polygon_cut.orthogonal_axis = 'z'
polygon_cut.axis_max = 10
polygon_cut.axis_min = -1
polygon_cut.bounding_polygon = utility.Vector3dVector([[0,0,0], [-5, -4, 0], [-5, 8, 0]])
pcd_cut = polygon_cut.crop_point_cloud(pcd)
print("pcd_cut shape: {}".format(np.asarray(pcd_cut.points).shape))

draw_geometries_dark_background([pcd])
draw_geometries_dark_background([pcd_cut])

# draw_geometries_dark_background([pcd, bbox, cylinder])
# draw_geometries_dark_background([pcd_cut, bbox, cylinder])
