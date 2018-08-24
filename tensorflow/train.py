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
    D0 = sio.loadmat('../samples/D0.mat')['D0']

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    save_img(X[:,:,2], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)

    tdsc = TDSC(m, n, k)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(params.sc_max_iter):
            time_start = time.time()
            print('Iteration: {} / {}'.format(i, params.sc_max_iter),)

            # compute tensor coefficients B
            sess.run(tdsc.B_assign, feed_dict={tdsc.X_p:X_p})

            # compute tensor dictionary D
            tdsc.dl_opt.minimize(sess, feed_dict={tdsc.X_p:X_p})
            sess.run(tdsc.D_assign, feed_dict={tdsc.X_p:X_p})

            # recover input tensor X
            sess.run(tdsc.B_assign, feed_dict={tdsc.X_p:X_p})
            X_p_recon = sess.run(tdsc.X_p_recon)
            X_recon = block_3d_tensor(X_p_recon, np.shape(X))

            save_img(X_recon[:, :, 2], './out/{}.png'.format(str(i).zfill(3)))

            time_end = time.time()
            print('time:', time_end - time_start, 's')

        # plt.imshow(X_recon[:,:,1])
        # plt.show()






