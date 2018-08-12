# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/12 14:40

import time
import scipy.io as sio
import numpy as np
from init import *
from block_3d import *
from tensor_dl import *
from tensor_tsta import *
from hyper_params import HyperParams as params
from matplotlib import pyplot as plt

def tdsc():
    X = sio.loadmat('samples/baloons_101_101_31.mat')['Omsi']
    size_X = np.shape(X)
    plt.imshow(X[:,:,1])
    plt.show()

    X_p = tensor_block_3d(X)
    size_X_p = np.shape(X_p)
    X_p_hat = np.fft.fft(X_p, axis=-1)

    D0 = init_3d_tensor(params.patch_size, params.r)
    B0 = np.zeros([params.r, size_X_p[1], size_X_p[2]])

    time_s = time.time()

    for i in range(params.sc_max_iter):
        print('Iteration: {} / {}'.format(i, params.sc_iter_num),)
        if i == 0:
            B = tensor_tsta(X_p, D0, B0)
        else:
            B = tensor_tsta(X_p, D, B)

        D = tensor_dl(X_p_hat, B, params.r)

    time_e = time.time()

    X_p_ = tensor_product(D, '', B, '')
    X_ = block_3d_tensor(X_p_, size_X)
    print('Total time:', time_e - time_s, 's')
    print(X[:,:,1])

    img = X_[:,:,1] * 255
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    #tensor = sio.loadmat('samples/baloons_101_101_31.mat')['Omsi']
    tdsc()
