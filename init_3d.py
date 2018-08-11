# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/10 21:03

import numpy as np
from params import Params as params

def init_3d():
    D_mat = np.random.rand(params.patch_size ** 3, params.r) * 2 - 1
    print(D_mat)
    D_mat_1 = np.sqrt(np.sum(np.square(D_mat), axis=0))
    for i in range(D_mat_1.shape[0]):
        D_mat[:,i] /= D_mat_1[i]
    D = np.transpose(np.reshape( \
        D_mat, [params.patch_size ** 2, params.patch_size, params.r]), [0, 2, 1])
    return D


if __name__ == '__main__':
    a = np.random.rand(10,12)
    print(a)
    print(a[0:2:10,:])
