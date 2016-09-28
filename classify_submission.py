#!/usr/bin/env python
# coding=utf-8
#Author: Perfe
#E-mail: ieqinglinzhang@gmail.com

import pandas as pd
import numpy as np

submission = pd.read_csv("./generated_data/2016_08_10_08_40_submission.csv")
test_class = np.load("./generated_data/test_class_v4.npy")
for num,i in enumerate(test_class):
    if i < 0.01:
        submission["pixels"][num] = ""

submission.to_csv("./generated_data/class_submission_06.csv",index=False)

