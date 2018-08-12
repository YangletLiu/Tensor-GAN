# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/12 14:40

import scipy.io as sio
import numpy as np
from init_3d import *
from block_3d import *
from tensor_dl import *
from tensor_tsta import *
from params import Params as params
from matplotlib import pyplot as plt

def tdsc():
    X = sio.loadmat('samples/baloons_101_101_31.mat')['Omsi']
    size_X = np.shape(X)
    X_p = tensor_block_3d(X)
    size_X_p = np.shape(X_p)
    X_p_hat = np.fft.fft(X_p, axis=-1)
    D0 = init_3d()
    B0 = np.zeros([params.r, size_X_p[1], size_X_p[2]])

    for i in range(params.sc_iter_num):
        print(i)
        if i == 0:
            B = tensor_tsta(X_p, D0, B0)
        else:
            B = tensor_tsta(X_p, D, B)

        D = tensor_dl(X_p_hat, B, params.r)


    X_p_ = tensor_product(D, '', B, '')
    X_ = block_3d_tensor(X_p_, size_X)

    img = X_[:,:,1]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    #tensor = sio.loadmat('samples/baloons_101_101_31.mat')['Omsi']
    tdsc()
