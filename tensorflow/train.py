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
sys.path.append('../')

from tdsc import TDSC
from block_3d import *
from hyper_params import HyperParams as params


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


if __name__ == '__main__':
    X = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']
    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    save_img(X[:,:,2], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    tdsc = TDSC(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tdsc.train(sess, X_p, X, params.sc_max_iter)

    sess.close()






