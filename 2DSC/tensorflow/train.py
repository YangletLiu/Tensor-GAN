# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/19 19:17

import tensorflow as tf
import numpy as np
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import time
import sys
import keras

sys.path.append('../')

from tdsc import Tdsc
from block_3d import *
from hyper_params import HyperParams as params


def train_balloon():
    X = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    Tdsc.save_img(X[:,:,2], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    tdsc = Tdsc(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tdsc.train(sess, X_p, X, params.sc_max_iter)

    sess.close()


def train_cifar10():
    (train_data, _), (_, _) = keras.datasets.cifar10.load_data()
    train_data = (train_data.astype(np.float32) - 127.5) / 127.5
    index = np.random.randint(0, train_data.shape[0])
    print(index)
    X = train_data[index]
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    Tdsc.save_img(X, './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    tdsc = Tdsc(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(params.sc_max_iter):
        X_recon = tdsc.train(sess, X_p, X, 1)
        Tdsc.save_img(X_recon, './out/{}.png'.format(str(i).zfill(3)))

    sess.close()


if __name__ == '__main__':
    # train_balloon()
    train_cifar10()






