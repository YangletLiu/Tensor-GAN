# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/12 14:40

import time
import os
import keras
import scipy.io as sio
import numpy as np
from init import *
from block_3d import *
from tensor_dl import *
from tensor_tsta import *
from tensor_product import *
from hyper_params import HyperParams as hp
from matplotlib import pyplot as plt


def save_img(img, file_name):
    fig = plt.figure(figsize=(5, 15))
    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    # plt.imshow(img, cmap='Greys_r')
    plt.imshow(img)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def tdsc(X_p):

    size_X_p = np.shape(X_p)
    X_p_hat = np.fft.fft(X_p, axis=-1)

    D = init_D(hp.patch_size, hp.r)
    C = np.zeros([hp.r, size_X_p[1], size_X_p[2]])

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    for i in range(hp.sc_max_iter):
        time_start = time.time()
        print('Iteration: {} / {}'.format(i, hp.sc_max_iter))

        C = tensor_tsta(X_p, D, C)
        D = tensor_dl(X_p_hat, C, hp.r)

        C = tensor_tsta(X_p, D, C)

        time_end = time.time()
        print('time:', time_end - time_start, 's')

    return D


def sample_cifar10(n_samples=10000):
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    X_p = np.zeros((hp.patch_size*hp.patch_size, n_samples, hp.patch_size))
    count = 0
    #蓄水池抽样算法
    for i in range(len(x_train)):
        img_p = tensor_block_3d(x_train[i])
        cur_count = img_p.shape[1]
        for j in range(cur_count):
            if count < n_samples:
                X_p[:,count,:] = img_p[:,j,:]
            else:
                k = np.random.randint(0, count)
                if k < n_samples:
                    X_p[:,k,:] = img_p[:,j,:]
            count += 1

    idx = np.random.randint(0, len(x_train))
    img = x_train[idx]
    img_p = tensor_block_3d(img)

    return X_p, img, img_p


if __name__ == '__main__':
    # X = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']
    X_p, img, img_p = sample_cifar10()
    D = tdsc(X_p)
    C = np.zeros((hp.r, img_p.shape[1], img_p.shape[2]))
    C = tensor_tsta(img_p, D, C)
    img_p_ = tensor_product(D, '', C, '')
    img_ = block_3d_tensor(img_p_, img.shape)
    img = img*0.5 + 0.5
    img_ = img_*0.5 + 0.5
    save_img(img, './out/origin.png')
    save_img(img_, './out/recon.png')
