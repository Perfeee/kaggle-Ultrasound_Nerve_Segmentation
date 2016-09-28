#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import cv2

train_images = np.load("./generated_data/test_images.npy")
train_images_eq = np.empty_like(train_images,dtype=np.uint8)
for i in range(train_images.shape[0]):
    image = train_images[i,0]
    image_eq = cv2.equalizeHist(image)
    train_images_eq[i,0] = image_eq
    print(i,"th pic was equalized")
print(train_images_eq.shape)
np.save("./generated_data/test_images_equalizedHist.npy",train_images_eq)
cv2.imshow("image",train_images[0,0])
cv2.imshow("image_eq",train_images_eq[0,0])
cv2.waitKey()
cv2.destroyAllWindows()
