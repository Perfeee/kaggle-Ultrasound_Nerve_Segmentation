#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import cv2

train_images = np.load("./generated_data/test_images.npy")
train_images_filter = np.empty_like(train_images,dtype=np.uint8) 

for i in range(train_images.shape[0]):
    newimg = cv2.medianBlur(train_images[i,0],3)
    train_images_filter[i,0] = newimg

print(train_images_filter.shape)
np.save("./generated_data/test_images_filter.npy",train_images_filter)
