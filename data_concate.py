#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
masks = np.load("./generated_data/train_masks_concate_2_64*80.npy")
masks_rotation = np.load("./generated_data/train_masks_64*80.npy")

images = np.load("./generated_data/train_images_concate_2_64*80.npy")
images_rotation = np.load("./generated_data/train_images_filter_64*80.npy")

masks_concate = np.concatenate((masks,masks_rotation),axis=0)
images_concate = np.concatenate((images,images_rotation),axis=0)
print(images_concate.shape)
print(masks_concate.shape)

np.save("./generated_data/train_images_concate_3_64*80.npy",images_concate)
np.save("./generated_data/train_masks_concate_3_64*80.npy",masks_concate)

