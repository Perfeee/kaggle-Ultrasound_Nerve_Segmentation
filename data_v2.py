#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function

import os
import numpy as np
import cv2

import glob

img_rows = 420
img_cols = 580

def create_train_data():
    train_data_path = glob.glob("../../train/*[0-9].tif")
#    train_mask_path = glob.glob("../../train/1_*mask.tif")
    
    train_data_path = np.array(train_data_path)
    np.save("./generated_data/train_data_list.npy",train_data_path)

    total = len(train_data_path)

    train_images = np.ndarray((total,1,img_rows,img_cols),dtype=np.uint8)
    train_class = np.ndarray((total,1),dtype=np.uint8)


    print("-"*38)
    print("Creating training images...")
    print("-"*30)
    for i,image_path in enumerate(train_data_path):
        mask_path = image_path[:-4] + "_mask.tif"

        image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        mask = cv2.imread(mask_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)

        image = np.array([image])
        mask = np.array([mask])
        if len(np.where(mask==255)[0]) > 500:
            mask_class = 1
        else:
            mask_class = 0

        train_images[i] = image
        train_class[i] = mask_class
        
        if i % 100 == 0:
            print('Done:{0}/{1} images'.format(i,total))

    print("Loading Done.")
    print("train data's shape: ",train_images.shape)
    
    np.save('./generated_data/train_images_v2.npy',train_images)
    np.save('./generated_data/train_class_v2.npy',train_class)
    print("train data saved.")



if __name__ == '__main__':
    create_train_data()
