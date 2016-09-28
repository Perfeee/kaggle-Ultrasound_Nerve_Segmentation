#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com


from __future__ import print_function

import cv2
import numpy as np

img_rows = 64
img_cols = 80


def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0],imgs.shape[1],img_rows,img_cols),dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i,0] = cv2.resize(imgs[i,0],(img_cols,img_rows),interpolation=cv2.INTER_CUBIC)
        print(i,"th pic was resized.")
    return imgs_p



def img_resize():
    
    print("-"*30)
    print("Loading train data...")
    print('-'*30)
    '''    
    train_images_rotation = np.load("./generated_data/train_images_rotation_2.npy")
    train_masks_rotation = np.load("./generated_data/train_masks_rotation_2.npy")

    train_images_rotation = preprocess(train_images_rotation)
    train_masks_rotation = preprocess(train_masks_rotation)
    '''
    train_images = np.load("./generated_data/train_masks_shift.npy")
    train_images = preprocess(train_images)
    print(train_images.shape)
    np.save("./generated_data/train_masks_shift_64*80.npy",train_images)


if __name__ == '__main__':
    img_resize()
