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
    
    

    total = len(train_data_path)

    train_images = np.ndarray((total,1,img_rows,img_cols),dtype=np.uint8)
    train_masks = np.ndarray((total,1,img_rows,img_cols),dtype=np.uint8)


    print("-"*38)
    print("Creating training images...")
    print("-"*30)
    for i,image_path in enumerate(train_data_path):
        mask_path = image_path[:-4] + "_mask.tif"

        image = cv2.imread(image_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)
        mask = cv2.imread(mask_path,cv2.CV_LOAD_IMAGE_GRAYSCALE)

        image = np.array([image])
        mask = np.array([mask])
        
        train_images[i] = image
        train_masks[i] = mask
        
        if i % 100 == 0:
            print('Done:{0}/{1} images'.format(i,total))

    print("Loading Done.")
    print("train data's shape: ",train_images.shape)
    
    np.save('./generated_data/train_images.npy',train_images)
    np.save('./generated_data/train_masks.npy',train_masks)
    print("train data saved.")

def load_train_data():
    train_images = np.load("./generated_data/train_images.npy")
    train_masks = np.load("./generated_data/train_masks.npy")
    return train_images,train_masks


def create_test_data():
    test_data_path = glob.glob("../../test/*[0-9].tif")

    total = len(test_data_path)

    test_images = np.ndarray((total,1,img_rows,img_cols),dtype=np.uint8)
    test_ids = np.ndarray((total,),dtype=np.int32)

    print("Createing test_data...")

    for i,image in enumerate(test_data_path):
        test_id = int(image.split('/')[-1].split('.')[0])
        test_image = cv2.imread(image,cv2.CV_LOAD_IMAGE_GRAYSCALE)

        test_image = np.array(test_image)
        
        test_images[i] = test_image
        test_ids[i] = test_id

        if i % 100 == 0:
            print("Done: {}/{}".format(i,total))

    print("test data's loading done.")

    np.save("./generated_data/test_images.npy",test_images)
    np.save("./generated_data/test_ids.npy",test_ids)
    print("test_data saved.")


def load_test_data():
    test_images = np.load('./generated_data/test_images.npy')
    test_ids = np.load("./generated_data/test_ids.npy")
    return test_images,test_ids



if __name__ == '__main__':
    create_train_data()
    create_test_data()
