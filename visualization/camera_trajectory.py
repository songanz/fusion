# import cv2
import sys
root_dir = "/home/songanz/Documents/Git_repo/fusion/"  # change this for your own usage
sys.path.append(root_dir + "Open3D/build/lib") # NOTE! you'll have to adapt this for your file structure
from open3d import *
import numpy as np


def read_calib_file(calib_path):
    out = dict()
    line_num = 1
    for line in open(calib_path, 'r'):
        line = line.strip()
        if line == '' or line[0] == '#':
            continue
        if line_num <= 4:
            val = line.split(':')
            key_name = val[0].strip()
            val = np.asarray(val[-1].strip().split(' '), dtype='f8')
            line_num += 1
        else:
            key_name = line.split(' ')[0]
            val = line.split(' ')
            val = np.asarray(val[1:])
        if len(val) == 12:
            out[key_name] = val.reshape(3, 4)
        elif len(val) == 9:
            out[key_name] = val.reshape(3, 3)

    return out

seq = '0004'  # change this if you want different sequence
point_cloud = np.fromfile(
    root_dir + "data/kitti/tracking/training/velodyne/" + seq + "/000000.bin",
    dtype=np.float32).reshape(-1, 4)
point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]

calib_path = root_dir + "data/kitti/tracking/training/calib/" + seq + ".txt"
calib = read_calib_file(calib_path)

point_cloud = point_cloud[point_cloud[:, 0] > -2.5, :]
point_cloud_xyz = point_cloud[:, 0:3]

P2 = calib['P2']
Tr_velo_to_cam_original = calib['Tr_velo_cam']
R0_rect_original = calib['R_rect']

R0_rect = np.eye(4)
R0_rect[0:3, 0:3] = R0_rect_original
Tr_velo_to_cam = np.eye(4)
Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_original
point_cloud_xyz = point_cloud[:, 0:3]

point_cloud_xyz_homo = np.ones((point_cloud.shape[0], 4))
point_cloud_xyz_homo[:, 0:3] = point_cloud[:, 0:3]

point_cloud_camera_non_rec = np.dot(Tr_velo_to_cam, point_cloud_xyz_homo.T)
point_cloud_camera_rect = np.dot(R0_rect, point_cloud_camera_non_rec).T  # 4 channels, homogeneous coordinates
point_cloud_xyz_camera = np.zeros((point_cloud_camera_rect.shape[0], 3))  # 3 channels , cartesian coordinates
point_cloud_xyz_camera[:, 0] = point_cloud_camera_rect[:, 0] / point_cloud_camera_rect[:, 3]
point_cloud_xyz_camera[:, 1] = point_cloud_camera_rect[:, 1] / point_cloud_camera_rect[:, 3]
point_cloud_xyz_camera[:, 2] = point_cloud_camera_rect[:, 2] / point_cloud_camera_rect[:, 3]

pcd = PointCloud()
pcd.points = Vector3dVector(point_cloud_xyz_camera)
pcd.paint_uniform_color([0.7, 0.7, 0.7])

vis = Visualizer()
vis.create_window()
ctr = vis.get_view_control()
opt = vis.get_render_option()
opt.background_color = np.asarray([0, 0, 0])
vis.add_geometry(pcd)
vis.run()

# draw_geometries_dark_background([pcd])
param = ctr.convert_to_pinhole_camera_parameters()
# camera_trajectory = PinholeCameraTrajectory()
# camera_trajectory.intrinsic = param[0]
# camera_trajectory.extrinsic = Matrix4dVector([param[1]])
write_pinhole_camera_parameters(root_dir + "visualization/camera_trajectory/" + seq + ".json", param)
vis.destroy_window()