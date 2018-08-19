# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/12 14:40

import time
import os
import scipy.io as sio
import numpy as np
from init import *
from block_3d import *
from tensor_dl import *
from tensor_tsta import *
from hyper_params import HyperParams as params
from matplotlib import pyplot as plt


def save_img(img, file_name):
    fig = plt.figure(figsize=(5, 15))
    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(img, cmap='Greys_r')
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def tdsc(X):
    size_X = np.shape(X)

    X_p = tensor_block_3d(X)
    size_X_p = np.shape(X_p)
    X_p_hat = np.fft.fft(X_p, axis=-1)

    D = init_D(params.patch_size, params.r)
    # D = sio.loadmat('./samples/D0.mat')['D0']
    B = np.zeros([params.r, size_X_p[1], size_X_p[2]])

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    save_img(X[:,:,2], './out/origin.png')

    for i in range(params.sc_max_iter):
        time_s = time.time()
        print('Iteration: {} / {}'.format(i, params.sc_max_iter))

        B = tensor_tsta(X_p, D, B)
        D = tensor_dl(X_p_hat, B, params.r)

        B = tensor_tsta(X_p, D, B)

        X_p_ = tensor_product(D, '', B, '')
        X_ = block_3d_tensor(X_p_, size_X)
        save_img(X_[:,:,2], './out/{}.png'.format(str(i).zfill(3)))

        time_e = time.time()
        print('time:', time_e - time_s, 's')

    #X_p_ = tensor_product(D, '', B, '')
    #X_ = block_3d_tensor(X_p_, size_X)

    #img = X_[:,:,1]
    #plt.imshow(img)
    #plt.show()


if __name__ == '__main__':
    X = sio.loadmat('samples/baloons_101_101_31.mat')['Omsi']
    tdsc(X)
