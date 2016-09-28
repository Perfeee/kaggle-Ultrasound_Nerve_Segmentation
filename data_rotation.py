#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import cv2

train_images = np.load("./generated_data/train_images.npy")
train_masks = np.load("./generated_data/train_masks.npy")
w = train_images[0,0].shape[0]
h = train_images[0,0].shape[1]
print((w,h))
rotation_center = (w/2,h/2)
rotation_angle = 10
rotation_scale = 1
transform_matrix = cv2.getRotationMatrix2D(center=rotation_center,angle=rotation_angle,scale=rotation_scale)

#train_images_rotation = np.empty_like(train_images,dtype=np.uint8) 
train_masks_rotation = np.empty_like(train_images,dtype=np.uint8) 

for i in range(train_images.shape[0]):
#    newimg = cv2.warpAffine(train_images[i,0],transform_matrix,(h,w))
    newmask = cv2.warpAffine(train_masks[i,0],transform_matrix,(h,w))
#    print(newimg.shape)
#    cv2.imshow("old",train_images[i,0])
#    cv2.imshow("newimg",newimg)
#    train_images_rotation[i,0] = newimg
    train_masks_rotation[i,0] = newmask

#print(train_images_rotation.shape)
print(train_masks_rotation.shape)
#np.save("./generated_data/train_images_rotation.npy",train_images_rotation)
np.save("./generated_data/train_masks_rotation.npy",train_masks_rotation)
