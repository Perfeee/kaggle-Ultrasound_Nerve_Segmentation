#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com
from __future__ import print_function
import numpy as np
train_masks = np.load("./generated_data/train_masks_64*80.npy")
train_images = np.load("./generated_data/train_images_64*80.npy")

pos = []
for i in range(5635):
    if len(np.where(train_masks[i,0]==255)[0]) > 0:
        pos.append(i)

train_images_passive = np.ndarray((len(pos),1,64,80),dtype=np.uint8)
train_masks_passive = np.ndarray((len(pos),1,64,80),dtype=np.uint8)
for i,p in enumerate(pos):
    train_images_passive[i,0] = train_images[p,0]
    train_masks_passive[i,0] = train_masks[p,0]

np.save("./generated_data/train_masks_passive.npy",train_masks_passive)
np.save("./generated_data/train_images_passive.npy",train_images_passive)

print(train_masks_passive.shape)
print(train_images_passive.shape)
