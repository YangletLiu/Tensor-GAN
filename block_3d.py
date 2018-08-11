# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/10 21:41

import numpy as np
from params import Params as params

def block_3d(X):
    size = np.shape(X)
    total_patch_num = (np.floor((size[0] - params.patch_size)/params.step) + 1) * \
                      (np.floor((size[1] - params.patch_size)/params.step) + 1) * \
                      (np.floor((size[2] - params.patch_size)/params.step) + 1)

    Z = np.zeros(params.patch_size, params.patch_size, params.patch_size, total_patch_num)
    for i in range(params.patch_size):
        for j in range(params.patch_size):
            for k in range(params.patch_size):
                patch = X[i:]
