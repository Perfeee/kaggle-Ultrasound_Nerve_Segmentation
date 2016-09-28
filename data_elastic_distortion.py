#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com
from __future__ import print_function
import numpy as np
import pandas as pd
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

def elastic_transform(image,alpha,sigma,alpha_affine,random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square+square_size,[center_square[0]+square_size,center_square[1]-square_size],center_square-square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine,alpha_affine,size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1,pts2)
    image = cv2.warpAffine(image,M,shape_size[::-1],borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape)*2-1),sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) *2-1),sigma) * alpha
    dz = np.zeros_like(dx)

    x,y,z = np.meshgrid(np.arange(shape[1]),np.arange(shape[0]),np.arange(shape[2]))
    indices = np.reshape(y+dy,(-1,1)), np.reshape(x+dx,(-1,1)), np.reshape(z,(-1,1))
    return map_coordinates(image,indices,order=1,mode='reflect').reshape(shape)


train_images = np.load("./generated_data/train_images.npy")
train_masks = np.load("./generated_data/train_masks.npy")
train_images_t = np.empty_like(train_images[5000:])
train_masks_t = np.empty_like(train_masks[5000:])

for i in range(5000,train_images.shape[0]):
    im_merge = np.concatenate((train_images[i,0][...,None],train_masks[i,0][...,None]),axis=2)
    im_merge_t = elastic_transform(im_merge,im_merge.shape[1]*2,im_merge.shape[1]*0.08,im_merge.shape[1]*0.08)
    image_t = im_merge_t[...,0]
    mask_t = im_merge_t[...,1]
    train_images_t[i-5000,0] = image_t
    train_masks_t[i-5000,0] = mask_t
    print(i,"th pic was computed")

np.save("./generated_data/train_images_elastic_12.npy",train_images_t)
np.save("./generated_data/train_masks_elastic_12.npy",train_masks_t)
