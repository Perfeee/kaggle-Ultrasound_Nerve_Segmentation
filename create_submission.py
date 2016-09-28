#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function

import datetime
import numpy as np
import cv2
from data import img_cols,img_rows
from run_length_encode import rl_encode

def prep(img):
    img = img.astype('float32')
    img = cv2.threshold(img,0.5,1.,cv2.THRESH_BINARY)[1].astype(np.uint8)
    img = cv2.resize(img,(img_cols,img_rows))
    return img

def run_length_encode(img):
    from itertools import chain
    x = img.transpose().flatten()
    y = np.where(x>0)[0]
    if len(y) < 2000:
        return ''
    z = np.where(np.diff(y)>1)[0]
    start = np.insert(y[z+1],0,y[0])
    end = np.append(y[z],y[-1])
    length = end - start
    res = [[s+1,l+1] for s,l in zip(list(start),list(length))]
    res = list(chain.from_iterable(res))

    return ' '.join([str(r) for r in res])


def submission():
    from data import load_test_data
    test_images,test_ids = load_test_data()
    test_masks = np.load('./generated_data/test_masks_model3rd_1_shift_3_3.npy')


    argsort = np.argsort(test_ids)
    test_ids = test_ids[argsort]
    test_masks = test_masks[argsort]
    
#    test_masks_420_580 = np.ndarray((5635,1,420,580),dtype=np.uint8)

    total = test_masks.shape[0]
    ids = []
    rles = []

    for i in range(total):
        mask = test_masks[i,0]
        mask = prep(mask)
#        test_masks_420_580[i] = mask
    
#        rle = run_length_encode(mask)
        rle = rl_encode(mask)
        rles.append(rle)
        ids.append(test_ids[i])
        if i % 100 == 0:
            print('{}/{}'.format(i,total))

    first_row = 'img,pixels'
    file_name = "./generated_data/"+datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")+'_submission.csv' 

    with open(file_name,'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(ids[i]) + ',' + rles[i]
            f.write(s + '\n')
    
#    np.save('./generated_data/test_masks_420_580.npy',test_masks_420_580)
if __name__ == '__main__':
    submission()
