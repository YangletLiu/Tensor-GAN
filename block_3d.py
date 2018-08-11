# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/10 21:41

import numpy as np
from params import Params as params

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

    Y = np.transpose(np.reshape(Z, [psize**2, psize, total_patch_num]), [0,2,1])

    return Y

def block_3d_tensor(Y, size):



if __name__ == '__main__':
    X = np.random.rand(101,101,31)
    print(tensor_block_3d(X).shape)
