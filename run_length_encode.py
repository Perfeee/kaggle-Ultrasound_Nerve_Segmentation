#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com
import numpy as np

def rl_encode(img):
    x = img.transpose().flatten()
    x = x.astype('int64')
    length = len(np.where(x>0)[0])
    y = np.diff(x)
    z = np.where(y>0)[0]
    w = np.where(y<0)[0]
    if x[-1] > 0:
        w = np.append(w,[len(y)-1])
    if length < 2000:
        return ''
    position = z + 2
    length = w - z

    strings = ''
    for (pos,l) in zip(position,length):
        strings = strings + str(pos) + ' ' + str(l) + ' '
    return strings
