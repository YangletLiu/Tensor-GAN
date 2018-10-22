# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/22 18:59
import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage import transform
import scipy.io as sio
from GAN import *


if __name__ == '__main__':
    n_sample = 10
    n_slice = 7
    n_step = 1
    img_shape = (28, 28, 1)

    latent_dim = 128
    LAMBDA = 10
    SIGMA = 1e-3
    batch_size = 32

    g = GAN([14, 14, 1], latent_dim, LAMBDA, SIGMA, batch_size)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if tf.train.get_checkpoint_state('./backup/'):
        saver.restore(sess, './backup/')
        print('********Restore the latest trained parameters.********')
    else:
        raise Exception('Model does not exist!')

    zs = np.random.normal(size=(n_sample, latent_dim))
    gs = sess.run(g.g, feed_dict={g.z:zs})
    datas = np.zeros((n_sample, img_shape[0], img_shape[1], n_slice))
    for i in range(n_sample):
        for k in range(n_slice):
            data = transform.resize(np.squeeze(gs[i]), (img_shape[0], img_shape[1]))
            datas[i, :, 0:(img_shape[0] - n_slice + n_step * k), k] = data[:, (n_slice - n_step * k):img_shape[1]]

    sio.savemat('./data/mnist_test_28_28_7.mat', {'YY':datas})
