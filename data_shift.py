#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import cv2

#train_images = np.load("./generated_data/train_images.npy")
train_masks = np.load("./generated_data/train_masks.npy")
#train_images_shift = np.zeros_like(train_images,dtype=np.uint8) 
train_masks_shift = np.zeros_like(train_masks,dtype=np.uint8) 

for i in range(train_masks.shape[0]):
#    train_images_shift[i,0,0:-66,:] = train_images[i,0,66:,:]
    train_masks_shift[i,0,0:-66,:] = train_masks[i,0,66:,:]

print(train_masks_shift.shape)
#np.save("./generated_data/train_images_shift.npy",train_images_shift)
np.save("./generated_data/train_masks_shift.npy",train_masks_shift)
