# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/10 21:41

import numpy as np
from hyper_params import HyperParams as params


def tensor_block_3d(X):
    size = np.shape(X)
    psize = params.patch_size
    step = params.step
    total_patch_num = int((np.floor((size[0] - psize)/step) + 1) * \
                      (np.floor((size[1] - psize)/step) + 1) * \
                      (np.floor((size[2] - psize)/step) + 1))
    Z = np.zeros([psize, psize, psize, total_patch_num])
    for i in range(psize):
        for j in range(psize):
            for k in range(psize):
                patch = X[i:size[0]-psize+i+1,j:size[1]-psize+j+1,k:size[2]-psize+k+1][::step,::step,::step]
                Z[i,j,k,:] = np.reshape(patch, [1, total_patch_num])

    X_p = np.transpose(np.reshape(Z, [psize**2, psize, total_patch_num]), [0,2,1])

    return X_p


def block_3d_tensor(X_p, size):
    psize = params.patch_size
    step = params.step
    len_r = int(np.floor((size[0] - psize) / step) + 1)
    len_c = int(np.floor((size[1] - psize) / step) + 1)
    len_s = int(np.floor((size[2] - psize) / step) + 1)
    X = np.zeros(size)
    W = np.zeros(size)

    Z_p = np.reshape( \
        np.transpose(X_p, [0, 2, 1]), \
        [psize, psize, psize, np.shape(X_p)[1]])

    for i in range(psize):
        for j in range(psize):
            for k in range(psize):
                shape = np.shape(X[i:(len_r-1)*step+i+1,j:(len_c-1)*step+j+1,k:(len_s-1)*step+k+1][::step,::step,::step])
                X[i:(len_r-1)*step+i+1,j:(len_c-1)*step+j+1,k:(len_s-1)*step+k+1][::step,::step,::step] += \
                    np.reshape(Z_p[i,j,k,:], shape)
                W[i:(len_r-1)*step+i+1,j:(len_c-1)*step+j+1,k:(len_s-1)*step+k+1][::step,::step,::step] += \
                    np.ones(shape)

    X = X / W
    return X



if __name__ == '__main__':
    YY_p = np.random.rand(25,33614,5)
    size=[101,101,31]
    print(block_3d_tensor(YY_p, size).shape)



