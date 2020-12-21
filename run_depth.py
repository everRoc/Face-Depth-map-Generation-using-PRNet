import numpy as np
import scipy.io as sio
import os
import shutil
from skimage.io import imread, imsave
import cv2
import os
from glob import glob

from api import PRN
import utils.depth_image as DepthImage

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

prn = PRN(is_dlib = True, is_opencv = False)

# ------------- load data
image_folder = '/home/chang/dataset/A_face1'
save_folder = '/home/chang/dataset/A_depth'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

first_dir = os.listdir(image_folder)
for img in first_dir:
    # 二级目录绝对路径
    if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'JPG':
        path_image = image_folder + '/' + str(img)
        if int((img.split(".")[0]).split("_")[-2]) > 1:
        #if int((img.split(".")[0]).split("_")[0]):
        #if (img.split(".")[0]).split("(")[0] == 'zheng ':
            image = imread(path_image)
            image_shape = [image.shape[0], image.shape[1]]
            pos = prn.process(image, None, None, image_shape)
            #if all(pos==[None,None,None]):
            if pos is None:
                continue
            kpt = prn.get_landmarks(pos)
            
            # 3D vertices
            vertices = prn.get_vertices(pos)
            depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)
            #cv2.imshow('IMAGE', image[:,:,::-1])
            #cv2.imshow('DEPTH', depth_scene_map)
            #cv2.waitKey(3000)
            save_path = save_folder + '/' + str(img)
            imsave(save_path, depth_scene_map)
            #shutil.copyfile(source, deter)


'''
#types = ('*.jpg', '*.png')
image_path_list= []
for files in types:
    image_path_list.extend(glob(os.path.join(image_folder, files)))
    #image_path_list.extend(image_folder + files)
total_num = len(image_path_list)
path_image = './TestImages/0.jpg'

image = imread(path_image)
image_shape = [image.shape[0], image.shape[1]]

pos = prn.process(image, None, None, image_shape)

kpt = prn.get_landmarks(pos)

# 3D vertices
vertices = prn.get_vertices(pos)

depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

cv2.imshow('IMAGE', image[:,:,::-1])
cv2.imshow('DEPTH', depth_scene_map)
cv2.waitKey(3000)

'''


