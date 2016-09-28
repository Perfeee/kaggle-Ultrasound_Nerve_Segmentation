#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

from __future__ import print_function
import numpy as np
import glob
import json
filepath = glob.glob("./../../train/*[0-9].tif")
np.save("./generated_data/filepath.npy",np.array(filepath))

