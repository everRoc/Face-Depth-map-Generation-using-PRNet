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
#import numpy as np
import os
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp
from time import time
from PIL import Image

import cv2
from predictor import PosPrediction


os.environ['CUDA_VISIBLE_DEVICES'] = '2'

prn = PRN(is_dlib = True, is_opencv = False)
prefix = '.'

if True:
    import dlib
    detector_path = os.path.join(prefix, 'Data/net-data/mmod_human_face_detector.dat')
    face_detector = dlib.cnn_face_detection_model_v1(
            detector_path)

def dlib_detect(image):
        return face_detector(image, 1)

# ------------- load data
image_folder = '/home/chang/dataset/B_test'
save_folder = '/home/chang/dataset/B_face1'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

resolution_inp = 256
resolution_op = 256

first_dir = os.listdir(image_folder)
for img in first_dir:
    # 二级目录绝对路径
    if img.split(".")[-1] == 'jpg' or img.split(".")[-1] == 'JPG':
        path_image = image_folder + '/' + str(img)
        #if int((img.split(".")[0]).split("_")[-2]) == 1:


        image = imread(path_image)
        image_shape = [image.shape[0], image.shape[1]]

        save_path = save_folder + '/' + str(img)

        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])

        detected_faces = dlib_detect(image)
        if len(detected_faces) == 0:###########################################
            print('warning: no detected face')
            #return None
            continue

        d = detected_faces[0].rect ## only use the first detected face (assume that each input image only contains one face)
        left = d.left(); right = d.right(); top = d.top(); bottom = d.bottom()
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size*0.14])
        #size = int(old_size*1.58)
        #size = int(old_size*1.4)
        size = int(old_size*1.2)
        print(left, right, top, bottom)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,resolution_inp - 1], [resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        image = image/255.
        cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))


        ## test cropped_image
        cropped_image = np.array(cropped_image*255.0, np.uint8)
        tmp_pil = Image.fromarray(cropped_image)
        tmp_pil.save('test_face_haven.jpg')

        imsave(save_path, cropped_image)




        #pos = prn.process(image, None, None, image_shape, save_path)



        #kpt = prn.get_landmarks(pos)

        # 3D vertices
        #vertices = prn.get_vertices(pos)

        #depth_scene_map = DepthImage.generate_depth_image(vertices, kpt, image.shape, isMedFilter=True)

        #cv2.imshow('IMAGE', image[:,:,::-1])
        #cv2.imshow('DEPTH', depth_scene_map)
        #cv2.waitKey(3000)




        #shutil.copyfile(source, deter)
