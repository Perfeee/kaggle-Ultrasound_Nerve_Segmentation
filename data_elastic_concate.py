#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import glob

filepath = glob.glob("./generated_data/train_masks_elastic_*.npy")

train_images_elastic = np.concatenate([np.load(i) for i in filepath],axis=0)
print(train_images_elastic.shape)
np.save("./generated_data/train_masks_elastic.npy",train_images_elastic)
