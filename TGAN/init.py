# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/10 21:03

import numpy as np
from hyper_params import HyperParams as params


def init_D(patch_size, r):
    D_mat = np.random.rand(patch_size ** 3 * 5, r) * 2 - 1
    D_mat_sum = np.sqrt(np.sum(np.square(D_mat), axis=0))
    for i in range(D_mat_sum.shape[0]):
        D_mat[:,i] /= D_mat_sum[i]
    D = np.transpose(np.reshape( \
        D_mat, [patch_size ** 2 * 5, patch_size, params.r], order='F'), [0, 2, 1])
    return D


if __name__ == '__main__':
    print(init_D(params.patch_size, params.r))
