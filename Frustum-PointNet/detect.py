# camera-ready
'''
Use the yolov3 2D boundbox to get the frustum points and run the frustum detection
'''

from __future__ import division

from datasets import EvalSequenceDatasetFrustumPointNet, wrapToPi, getBinCenter # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)
from frustum_pointnet import FrustumPointNet

import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
import os
root_dir = "/home/songanz/Documents/Git_repo/fusion/"  # change this for your own usage
sys.path.append(root_dir + "utils")
from kittiloader import LabelLoader2D3D, calibread, LabelLoader2D3D_sequence # (this needs to be imported before torch, because cv2 needs to be imported before torch for some reason)

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages/')

sys.path.append(root_dir)
from yolov3.models import *
from utils.utils_yolo import *
from yolov3.datasets import *

sys.path.append(root_dir + "Open3D/build/lib")
import open3d

############### Set up path ##########################
F_weights = root_dir + "pretrained_models/model_37_2_epoch_400.pth"

Y_weights = root_dir + 'pretrained_models/yolov3/yolov3-kitti.weights'
Y_cfgfile = root_dir + 'config/yolov3-kitti.cfg'

sequence = "0004"
kitti_data_dir=root_dir + "data/kitti"
kitti_meta_dir=root_dir + "data/kitti/meta"
img_dir = kitti_data_dir + "/tracking/training/image_02/" + sequence + "/"
lidar_dir = kitti_data_dir + "/tracking/training/velodyne/" + sequence + "/"
label_path = kitti_data_dir + "/tracking/training/label_02/" + sequence + ".txt"
calib_path = kitti_meta_dir + "/tracking/training/calib/" + sequence + ".txt"  # NOTE! the data format for the calib files was sliightly different for tracking --> revised ones in meta folder

img_list = os.listdir(img_dir)
img_list.sort()
lidar_list = os.listdir(lidar_dir)
lidar_list.sort()

############# Set up useful constant #################
classes = load_classes(root_dir + 'data/kitti.names')
# for visualization Yolov3
cmap = plt.get_cmap('tab20b')
colors = [cmap(i) for i in np.linspace(0, 1, 20)]
cvfont = cv2.FONT_HERSHEY_PLAIN

count_empty = 0  # for count how many 2D detection end up with no 3D detection
pre_frame_pred_seg = {}  # previous frame predection, stored for tracking
cur_frame_pred_seg = {}
threshold_icp = 0.05

################ Load Networks #######################
F_network = FrustumPointNet("Frustum-PointNet_eval_val_seq", project_dir=root_dir)
F_network.load_state_dict(torch.load(F_weights))
F_network = F_network.cuda()
NH = F_network.BboxNet_network.NH

Y_network = Darknet(Y_cfgfile)
Y_network.load_weights(Y_weights)
Y_network.cuda()

F_network.eval() # (set in evaluation mode, this affects BatchNorm, dropout etc.)
Y_network.eval()

################ Useful functions ####################
def resize_img(img, img_size=416):
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    # Add padding
    input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
    padded_h, padded_w, _ = input_img.shape
    # Resize and normalize
    input_img = resize(input_img, (img_size, img_size, 3), mode='reflect')
    # Channels-first
    input_img = np.transpose(input_img, (2, 0, 1))
    # As pytorch tensor
    input_img = torch.from_numpy(input_img).float().unsqueeze(0)
    return input_img, img


for frame in range(len(os.listdir(img_dir))):
    img_path = img_dir + img_list[frame]
    img = np.array(Image.open(img_path))

    resized_img, img = resize_img(img, img_size=416)
    input_img = Variable(resized_img.type(torch.cuda.FloatTensor))

    prev_time = time.time()  # for calculate fps

    with torch.no_grad():  # (corresponds to setting volatile=True in all variables, this is done during inference to reduce memory consumption)
        BB_2D = Y_network(input_img)
        alldetections = non_max_suppression(BB_2D, 80, 0.8, 0.4)

    detection_Y = alldetections[0]

    # calcualte fps for yolo
    current_time_yolo = time.time()
    inference_time_yolo = current_time_yolo - prev_time
    fps_yolo = int(1 / inference_time_yolo)
    print ("fps_yolov3: ", fps_yolo)

    kitti_img_size = 416
    # The amount of padding that was added
    pad_x = max(img.shape[0] - img.shape[1], 0) * (kitti_img_size / max(img.shape))
    pad_y = max(img.shape[1] - img.shape[0], 0) * (kitti_img_size / max(img.shape))

    # Image height and width after padding is removed
    unpad_h = kitti_img_size - pad_y
    unpad_w = kitti_img_size - pad_x

    cur_frame_pred_seg = {}

    ######################## Debug visulization  #######################
    # if detection is not None:
    #     # print(img.shape)
    #     unique_labels = detection[:, -1].cpu().unique()
    #     n_cls_preds = min(len(unique_labels), 20)
    #     bbox_colors = random.sample(colors, n_cls_preds)
    #     for x1, y1, x2, y2, conf, cls_conf, cls_pred in detection:
    #         cls_pred = min(cls_pred, 8)  # refer to kitti.names
    #         print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
    #         # Rescale coordinates to original dimensions
    #         box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
    #         box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
    #         y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))
    #         x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))
    #         x2 = int(x1 + box_w)
    #         y2 = int(y1 + box_h)
    #         color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
    #         # print(color)
    #         cv2.line(img, (int(x1), int(y1 - 5)), (int(x2), int(y1 - 5)), (255, 255, 255), 14)
    #         cv2.putText(img, classes[int(cls_pred)], (int(x1), int(y1)), cvfont, 1.5,
    #                     (color[0] * 255, color[1] * 255, color[2] * 255), 2)
    #         cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (color[0] * 255, color[1] * 255, color[2] * 255),
    #                       1)
    # cv2.imshow('frame', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    ####################################################################

    # for each bounding box in each frame
    for b_indx, [x1, y1, x2, y2, conf, cls_conf, cls_pred] in enumerate(detection_Y):
        # Rescale coordinates to original dimensions
        cls_pred = min(cls_pred, 8)  # refer to kitti.names
        print ('\t+ Label: %s, Conf: %.5f' % (classes[int(cls_pred)], cls_conf.item()))
        box_h = int(((y2 - y1) / unpad_h) * (img.shape[0]))
        box_w = int(((x2 - x1) / unpad_w) * (img.shape[1]))
        y1 = int(((y1 - pad_y // 2) / unpad_h) * (img.shape[0]))  # minimum
        x1 = int(((x1 - pad_x // 2) / unpad_w) * (img.shape[1]))  # minimum
        x2 = int(x1 + box_w)
        y2 = int(y1 + box_h)

        ###################Get frustum points in the bounding box####################
        # first use the calib to get the lidar points in image coordinate
        lidar_path = lidar_dir + lidar_list[frame]
        point_cloud_ori = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)

        # remove points that are located behind the camera:
        point_cloud = point_cloud_ori[point_cloud_ori[:, 0] > 0, :]
        # remove points that are located too far away from the camera:
        point_cloud = point_cloud[point_cloud[:, 0] < 80, :]

        # Get calibration
        calib = calibread(calib_path)
        P2 = calib["P2"]
        Tr_velo_to_cam_orig = calib["Tr_velo_to_cam"]  # Rigid transformation from Velodyne to (non-rectified) camera coordinates
        R0_rect_orig = calib["R0_rect"]
        #
        R0_rect = np.eye(4)
        R0_rect[0:3, 0:3] = R0_rect_orig
        #
        Tr_velo_to_cam = np.eye(4)
        Tr_velo_to_cam[0:3, :] = Tr_velo_to_cam_orig

        point_cloud_xyz = point_cloud[:, 0:3]
        point_cloud_xyz_hom = np.ones((point_cloud.shape[0], 4))
        point_cloud_xyz_hom[:, 0:3] = point_cloud[:, 0:3]  # (point_cloud_xyz_hom has shape (num_points, 4))

        # project the points onto the image plane (homogeneous coords):
        img_points_hom = np.dot(P2, np.dot(R0_rect,
                                           np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T))).T  # (point_cloud_xyz_hom.T has shape (4, num_points))

        # normalize: for calculate row mask
        img_points = np.zeros((img_points_hom.shape[0], 2))
        img_points[:, 0] = img_points_hom[:, 0] / img_points_hom[:, 2]
        img_points[:, 1] = img_points_hom[:, 1] / img_points_hom[:, 2]

        # transform the points into (rectified) camera (0) coordinates:
        point_cloud_xyz_camera_hom = np.dot(R0_rect,
                                            np.dot(Tr_velo_to_cam, point_cloud_xyz_hom.T)).T  # (point_cloud_xyz_hom.T has shape (4, num_points))

        # normalize:
        point_cloud_xyz_camera = np.zeros((point_cloud_xyz_camera_hom.shape[0], 3))
        point_cloud_xyz_camera[:, 0] = point_cloud_xyz_camera_hom[:, 0] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 1] = point_cloud_xyz_camera_hom[:, 1] / point_cloud_xyz_camera_hom[:, 3]
        point_cloud_xyz_camera[:, 2] = point_cloud_xyz_camera_hom[:, 2] / point_cloud_xyz_camera_hom[:, 3]

        point_cloud_camera = point_cloud
        point_cloud_camera[:, 0:3] = point_cloud_xyz_camera

        # get the rwo mask using bounding box --> back to original point cloud
        row_mask = np.logical_and(
            np.logical_and(img_points[:, 0] >= x1,
                           img_points[:, 0] <= x2),
            np.logical_and(img_points[:, 1] >= y1,
                           img_points[:, 1] <= y2))

        frustum_point_cloud_xyz = point_cloud_xyz[row_mask, :]  # (needed only for visualization)
        frustum_point_cloud = point_cloud[row_mask, :]
        frustum_point_cloud_xyz_camera = point_cloud_xyz_camera[row_mask, :]
        frustum_point_cloud_camera = point_cloud_camera[row_mask, :]

        if frustum_point_cloud.shape[0] < 1024:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=True)
        else:
            row_idx = np.random.choice(frustum_point_cloud.shape[0], 1024, replace=False)

        frustum_point_cloud_xyz = frustum_point_cloud_xyz[row_idx, :]
        frustum_point_cloud = frustum_point_cloud[row_idx, :]
        frustum_point_cloud_xyz_camera = frustum_point_cloud_xyz_camera[row_idx, :]
        frustum_point_cloud_camera = frustum_point_cloud_camera[row_idx, :]

        # get centered_frustum_point_clouds which shoud be the input of F_network
        x_center = (x2 - x1)/2 + x1
        y_center = (y2 - y1)/2 + y1
        center_img_point_hom = np.array([x_center, y_center, 1])

        # (more than one 3D point is projected onto the center image point, i.e,
        # the linear system of equations is under-determined and has inf number
        # of solutions. By using the pseudo-inverse, we obtain the least-norm sol)

        # get a point (the least-norm sol.) that projects onto the center image point, in hom. coords:
        P2_pseudo_inverse = np.linalg.pinv(P2)  # (shape: (4, 3)) (P2 has shape (3, 4))
        point_hom = np.dot(P2_pseudo_inverse, center_img_point_hom)

        # hom --> normal coords:
        point = np.array(([point_hom[0] / point_hom[3], point_hom[1] / point_hom[3], point_hom[2] / point_hom[3]]))

        # if the point is behind the camera, switch to the mirror point in front of the camera:
        if point[2] < 0:
            point[0] = -point[0]
            point[2] = -point[2]

        # compute the angle of the point in the x-z plane: ((rectified) camera coords)
        frustum_angle = np.arctan2(point[0],
                                   point[2])  # (np.arctan2(x, z)) # (frustum_angle = 0: frustum is centered)

        # rotation_matrix to rotate points frustum_angle around the y axis (counter-clockwise):
        frustum_R = np.asarray([[np.cos(frustum_angle), 0, -np.sin(frustum_angle)],
                                [0, 1, 0],
                                [np.sin(frustum_angle), 0, np.cos(frustum_angle)]],
                               dtype='float32')

        # rotate the frustum point cloud to center it:
        centered_frustum_point_cloud_xyz_camera = np.dot(frustum_R, frustum_point_cloud_xyz_camera.T).T

        # subtract the centered frustum train xyz mean:
        with open(kitti_meta_dir + "/kitti_centered_frustum_mean_xyz.pkl", "rb") as file: # (needed for python3)
            centered_frustum_mean_xyz = pickle.load(file)
            centered_frustum_mean_xyz = centered_frustum_mean_xyz.astype(np.float32)

        centered_frustum_point_cloud_xyz_camera -= centered_frustum_mean_xyz

        centered_frustum_point_cloud_camera = frustum_point_cloud_camera
        centered_frustum_point_cloud_camera[:, 0:3] = centered_frustum_point_cloud_xyz_camera

        # for input to F_network
        centered_frustum_point_cloud_camera = centered_frustum_point_cloud_camera[np.newaxis, :, :]
        centered_frustum_point_cloud_camera = torch.from_numpy(centered_frustum_point_cloud_camera) # (shape: (1024, 4))
        centered_frustum_point_cloud_camera = Variable(centered_frustum_point_cloud_camera).transpose(2, 1).cuda()

        # Detection from the F_network
        detection_F = F_network(centered_frustum_point_cloud_camera)  # centered_frustum_point_cloud_camera

        # get segmentation
        detection_InstanceSeg = detection_F[0][0].data.cpu().numpy()
        row_mask_pred = detection_InstanceSeg[:, 1] > detection_InstanceSeg[:, 0]
        pred_seg_point_cloud = frustum_point_cloud[row_mask_pred, :]

        # get the physical center of that segmentation
        detection_TNet = detection_F[1][0].data.cpu().numpy()
        detection_TNet += np.mean(pred_seg_point_cloud, axis=0)[0:3]

        ############################ Debug visulization  ############################
        #
        # axis = open3d.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
        #
        # # visualize all transfermation in order to make sure they are exactly what I want
        # # 1. point_cloud_ori
        # point_cloud_ori_viz = open3d.PointCloud()
        # point_cloud_ori_viz.points = open3d.Vector3dVector(point_cloud_ori[:, 0:3])
        # point_cloud_ori_viz.paint_uniform_color([0.25, 0.25, 0.25])  # black
        #
        # # 2. point_cloud_xyz
        # point_cloud_xyz_viz = open3d.PointCloud()
        # point_cloud_xyz_viz.points = open3d.Vector3dVector(point_cloud_ori[:, 0:3])
        # point_cloud_xyz_viz.paint_uniform_color([0, 0, 1])  # blue
        #
        # # 3. point_cloud_xyz_hom
        # point_cloud_xyz_hom_viz = open3d.PointCloud()
        # point_cloud_xyz_hom_viz.points = open3d.Vector3dVector(point_cloud_xyz_hom[:, 0:3])
        # point_cloud_xyz_hom_viz.paint_uniform_color([1, 0, 0])  # red
        #
        # # 4. img_points_hom
        # img_points_hom_viz = open3d.PointCloud()
        # img_points_hom_viz.points = open3d.Vector3dVector(img_points_hom[:, 0:3])
        # img_points_hom_viz.paint_uniform_color([0, 1, 0])  # green
        #
        # # 5. img_points cannot be visualized as point cloud
        #
        # # 6. point_cloud_xyz_camera_hom
        # point_cloud_xyz_camera_hom_viz = open3d.PointCloud()
        # point_cloud_xyz_camera_hom_viz.points = open3d.Vector3dVector(point_cloud_xyz_camera_hom[:, 0:3])
        # point_cloud_xyz_camera_hom_viz.paint_uniform_color([0, 1, 0])  # green
        #
        # # 7. point_cloud_camera
        # point_cloud_camera_viz = open3d.PointCloud()
        # point_cloud_camera_viz.points = open3d.Vector3dVector(point_cloud_camera[:, 0:3])
        # point_cloud_camera_viz.paint_uniform_color([1, 0, 0])  # red
        #
        # # 8. frustum_point_cloud
        # frustum_point_cloud_viz = open3d.PointCloud()
        # frustum_point_cloud_viz.points = open3d.Vector3dVector(frustum_point_cloud[:, 0:3])
        # frustum_point_cloud_viz.paint_uniform_color([0, 1, 0])  # green
        #
        # # 9. frustum_point_cloud_camera
        # frustum_point_cloud_camera_viz = open3d.PointCloud()
        # frustum_point_cloud_camera_viz.points = open3d.Vector3dVector(frustum_point_cloud_camera[:, 0:3])
        # frustum_point_cloud_camera_viz.paint_uniform_color([1, 0, 0])  # red
        #
        # # 10. centered_frustum_point_cloud_camera
        # centered_frustum_point_cloud_camera_viz = open3d.PointCloud()
        # centered_frustum_point_cloud_camera_viz.points = open3d.Vector3dVector(centered_frustum_point_cloud_xyz_camera[:, 0:3])
        # centered_frustum_point_cloud_camera_viz.paint_uniform_color([1, 0, 0])  # red
        #
        # # 11. pred_seg_point_cloud
        # pred_seg_point_cloud_viz = open3d.PointCloud()
        # pred_seg_point_cloud_viz.points = open3d.Vector3dVector(pred_seg_point_cloud[:, 0:3])
        # pred_seg_point_cloud_viz.paint_uniform_color([0.25, 0.25, 0.25])  # black
        #
        # # 12. detection_TNet
        # detection_TNet_viz = open3d.PointCloud()
        # detection_TNet_viz.points = open3d.Vector3dVector(detection_TNet[np.newaxis, :])
        # detection_TNet_viz.paint_uniform_color([0, 0, 1])  # blue
        #
        # open3d.draw_geometries([pred_seg_point_cloud_viz, frustum_point_cloud_viz, point_cloud_camera_viz, detection_TNet_viz, axis])
        #
        #############################################################################

        if pred_seg_point_cloud.size == 0:
            count_empty += 1
            continue

        distance_pred_seg = np.min(pred_seg_point_cloud[:,0]**2 + pred_seg_point_cloud[:,1]**2 + pred_seg_point_cloud[:,2]**2)
        distance_pred_seg = np.sqrt(distance_pred_seg)
        print ("frame: ", frame, "  b_indx: ", b_indx, "    distance: ", distance_pred_seg)
        print ("count empty: ", count_empty)

        cur_frame_pred_seg[b_indx] = [pred_seg_point_cloud, detection_TNet, distance_pred_seg]

        if pre_frame_pred_seg:
            dist = 100  # for calculate the distance between frames
            for scoure in pre_frame_pred_seg:
                center_pre = pre_frame_pred_seg[scoure][1]
                dist_temp = np.linalg.norm(center_pre-detection_TNet)

                if dist_temp < dist:
                    dist = dist_temp
                    source_id = scoure
                    range_pre = pre_frame_pred_seg[scoure][2]

            if dist < 3:
                range_rate = (distance_pred_seg - range_pre)*10

            print ("Range: ", distance_pred_seg, "  Range_pre: ", range_pre, "    Range_rate:", range_rate)

        ############################ Debug visulization  ############################
        #
        # current_pc = open3d.PointCloud()
        # current_pc.points = open3d.Vector3dVector(pred_seg_point_cloud[:, 0:3])
        #
        # if pre_frame_pred_seg:
        #     pre_pc = open3d.PointCloud()
        #     pre_pc.points = open3d.Vector3dVector(pre_frame_pred_seg[source_id][0][:, 0:3])
        #     # for source in pre_frame_pred_seg:
        #     #     source_pc = open3d.PointCloud()
        #     #     source_pc.points = open3d.Vector3dVector(pre_frame_pred_seg[source][0][:, 0:3])
        #         # reg_p2p = open3d.registration_icp(source_pc, target_pc, threshold_icp, np.eye(4),
        #         #                                   open3d.TransformationEstimationPointToPoint(),
        #         #                                   open3d.ICPConvergenceCriteria(max_iteration = 2000))
        #         # print (reg_p2p.transformation)
        #
        #     current_pc.paint_uniform_color([0, 1, 0])  # green
        #     pre_pc.paint_uniform_color([1, 0, 0])  # red
        #     axis = open3d.create_mesh_coordinate_frame(size=1, origin=[0, 0, 0])
        #     open3d.draw_geometries([current_pc, pre_pc, axis])
        #
        #############################################################################

    pre_frame_pred_seg = cur_frame_pred_seg

    # Calcualte fps for frustum_pointnet detection
    current_time_frustum = time.time()
    inference_time_frustum = current_time_frustum - current_time_yolo
    fps_frustum = int(1 / inference_time_frustum)
    print("fps_frustum: ", fps_frustum)
    print("fps_total: ", int(1 / (inference_time_frustum + inference_time_yolo)))
