import numpy as np
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages/')

import os
root_dir = "/home/songanz/Documents/Git_repo/fusion/"

# set your data_set absolute path
# as for me, for example
# test example
kitti_img_path_train = root_dir + 'data/KITTI/images/train/'
kitti_img_path_val = root_dir + 'data/KITTI/images/val/'

kitti_label_path_train = root_dir + 'data/KITTI/labels/train/'
kitti_label_path_val = root_dir + 'data/KITTI/labels/val/'

kitti_img_path_test = root_dir + 'data/KITTI/testing/image_2/'



#transformed lables to save path
kitti_label_tosave_path = 'KITTI/labels2coco/'

#the absolute ptah of your data set
#kitti_data_real_path = root_dir + 'data/kitti/images/train/'

index = 0
cvfont = cv2.FONT_HERSHEY_SIMPLEX

kitti_names = open('kitti.names','r')
kitti_names_contents = kitti_names.readlines()                
kitti_images_train = os.listdir(kitti_img_path_train)
kitti_images_val = os.listdir(kitti_img_path_val)
kitti_images_test = os.listdir(kitti_img_path_test)

kitti_labels_train = os.listdir(kitti_label_path_train)
kitti_labels_val = os.listdir(kitti_label_path_val)

kitti_images_train.sort()
kitti_images_val.sort()
kitti_images_test.sort()
kitti_labels_train.sort()
kitti_labels_val.sort()

kitti_names_dic_key = []
for class_name in kitti_names_contents:
    kitti_names_dic_key.append(class_name.rstrip())
values = range(len(kitti_names_dic_key))
kitti_names_num = dict(zip(kitti_names_dic_key,values))

#print(kitti_names_num)

#创建训练集图片的List
f = open('train.txt','w')
for img in kitti_images_train:
    f.write(kitti_img_path_train+img+'\n')
f.close()

g = open('val.txt', 'w')
for img in kitti_images_val:
    g.write(kitti_img_path_val+img+'\n')
g.close()

f = open('test.txt','w')
for img in kitti_images_test:
    f.write(kitti_img_path_test+img+'\n')
f.close()

#kitti数据集 相对坐标 转换为绝对坐标
# for train
for indexi in range(len(kitti_images_train)):
    kitti_img_totest_path = kitti_img_path_train + kitti_images_train[indexi]
    kitti_label_totest_path = kitti_label_path_train + kitti_labels_train[indexi]
    #print(kitti_label_totest_path,kitti_img_totest_path)

    kitti_img_totest = cv2.imread(kitti_img_totest_path)
    #print(kitti_img_totest,type(kitti_img_totest))
    img_height, img_width = kitti_img_totest.shape[0],kitti_img_totest.shape[1]

    kitti_label_totest = open(kitti_label_totest_path,'r')

    label_contents = kitti_label_totest.readlines()
    #print(label_contents)
    real_label = open(kitti_label_tosave_path + kitti_labels_train[indexi],'w')

    for line in label_contents:
        data = line.split(' ')
        x=y=w=h=0
        if(len(data) == 15):
            class_str = data[0]
            if(class_str != 'DontCare'):
                # for kitti calls is a string
                # trans this to number by using kitti.names
                #(x,y) center (w,h) size
                x1 = float(data[4])
                y1 = float(data[5])
                x2 = float(data[6])
                y2 = float(data[7])

                intx1 = int(x1)
                inty1 = int(y1)
                intx2 = int(x2)
                inty2 = int(y2)

                bbox_center_x = float( (x1 + (x2 - x1) / 2.0) / img_width)
                bbox_center_y = float( (y1 + (y2 - y1) / 2.0) / img_height)
                bbox_width = float((x2 - x1) / img_width)
                bbox_height = float((y2 - y1) / img_height)

                #print(kitti_names_contents[class_num])
                # cv2.putText()
                # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
                #cv2.putText(kitti_img_totest, class_str, (intx1, inty1+3), cvfont, 2, (0,0,255), 1)
                # cv2.rectangle()
                # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                #cv2.rectangle(kitti_img_totest, (intx1,inty1), (intx2,inty2), (0,255,0), 2)
                line_to_write = str(kitti_names_num[class_str]) + ' ' + str(bbox_center_x)+ ' ' + str(bbox_center_y)+ ' ' + str(bbox_width)+ ' ' + str(bbox_height) +'\n'
                real_label.write(line_to_write)
                sys.stdout.write(str(int((indexi/len(kitti_images_train))*100))+'% '+'*******************->' "\r" )
                sys.stdout.flush()

    #cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)
    #cv2.waitKey()
    real_label.close()
# for test
for indexi in range(len(kitti_images_val)):
    kitti_img_totest_path = kitti_img_path_val + kitti_images_val[indexi]
    kitti_label_totest_path = kitti_label_path_val + kitti_labels_val[indexi]
    # print(kitti_label_totest_path,kitti_img_totest_path)

    kitti_img_totest = cv2.imread(kitti_img_totest_path)
    # print(kitti_img_totest,type(kitti_img_totest))
    img_height, img_width = kitti_img_totest.shape[0], kitti_img_totest.shape[1]

    kitti_label_totest = open(kitti_label_totest_path, 'r')

    label_contents = kitti_label_totest.readlines()
    # print(label_contents)
    real_label = open(kitti_label_tosave_path + kitti_labels_val[indexi], 'w')

    for line in label_contents:
        data = line.split(' ')
        x = y = w = h = 0
        if (len(data) == 15):
            class_str = data[0]
            if (class_str != 'DontCare'):
                # for kitti calls is a string
                # trans this to number by using kitti.names
                # (x,y) center (w,h) size
                x1 = float(data[4])
                y1 = float(data[5])
                x2 = float(data[6])
                y2 = float(data[7])

                intx1 = int(x1)
                inty1 = int(y1)
                intx2 = int(x2)
                inty2 = int(y2)

                bbox_center_x = float((x1 + (x2 - x1) / 2.0) / img_width)
                bbox_center_y = float((y1 + (y2 - y1) / 2.0) / img_height)
                bbox_width = float((x2 - x1) / img_width)
                bbox_height = float((y2 - y1) / img_height)

                # print(kitti_names_contents[class_num])
                # cv2.putText()
                # 输入参数为图像、文本、位置、字体、大小、颜色数组、粗细
                # cv2.putText(kitti_img_totest, class_str, (intx1, inty1+3), cvfont, 2, (0,0,255), 1)
                # cv2.rectangle()
                # 输入参数分别为图像、左上角坐标、右下角坐标、颜色数组、粗细
                # cv2.rectangle(kitti_img_totest, (intx1,inty1), (intx2,inty2), (0,255,0), 2)
                line_to_write = str(kitti_names_num[class_str]) + ' ' + str(bbox_center_x) + ' ' + str(
                    bbox_center_y) + ' ' + str(bbox_width) + ' ' + str(bbox_height) + '\n'
                real_label.write(line_to_write)
                sys.stdout.write(str(int((indexi / len(kitti_images_val)) * 100)) + '% ' + '*******************->' "\r")
                sys.stdout.flush()

    # cv2.imshow(str(indexi)+' kitti_label_show',kitti_img_totest)
    # cv2.waitKey()
    real_label.close()

kitti_names.close()
print("Labels tranfrom finished!")
