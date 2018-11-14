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
from conv_wgan_gp_mnist import *
from conv_wgan_gp_cifar10 import *


def mnist():
    n_samples = 32
    n_slice = 7
    n_step = 1
    img_shape = (28, 28, 1)

    learning_rate = 1e-4
    LAMBDA = 10
    step_num = 100000
    batch_size = 32

    g = ConvWganGpMnist(
        z_shape=100,
        batch_size=batch_size,
        step_num=step_num,
        learning_rate=learning_rate,
        LAMBDA=LAMBDA,
        DIM=64
    )

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if tf.train.get_checkpoint_state('./backup/mnist/'):
        saver.restore(sess, './backup/mnist/')
        print('********Restore the latest trained parameters.********')
    else:
        raise Exception('Model does not exist!')

    zs = sample_z(n_samples, g.z_shape)
    gms = sess.run(g.gm, feed_dict={g.z:zs})
    datas = np.zeros((n_samples, img_shape[0], img_shape[1], n_slice))
    for i in range(n_samples):
        for k in range(n_slice):
            data = transform.resize(np.squeeze(gms[i]), (img_shape[0], img_shape[1]))
            datas[i, :, 0:(img_shape[0] - n_slice + n_step * k), k] = data[:, (n_slice - n_step * k):img_shape[1]]

    sio.savemat('./data/mnist_test_14_14_7.mat', {'YY':datas})

def cifar10():
    n_samples = 32
    n_slice = 9
    n_step = 1
    img_shape = (32, 32, 3)

    learning_rate = 1e-4
    LAMBDA = 10
    step_num = 100000
    batch_size = 32

    g = ConvWganGpCifar10(
        z_shape=100,
        batch_size=batch_size,
        step_num=step_num,
        learning_rate=learning_rate,
        LAMBDA=LAMBDA,
        DIM=64
    )

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    if tf.train.get_checkpoint_state('./backup/cifar10/'):
        saver.restore(sess, './backup/cifar10/')
        print('********Restore the latest trained parameters.********')
    else:
        raise Exception('Model does not exist!')

    zs = sample_z(n_samples, g.z_shape)
    gms = sess.run(g.gm, feed_dict={g.z:zs})
    datas = np.zeros((n_samples, img_shape[0], img_shape[1], n_slice))
    for i in range(n_samples):
        for k in range(n_slice//3):
            data = transform.resize(np.squeeze(gms[i]), (img_shape[0], img_shape[1]))
            datas[i, :, :, (k*3):((k+1)*3)] = data

    sio.savemat('./data/cifar10_test_16_16_9.mat', {'YY':datas})


if __name__ == '__main__':
    mnist()
    # cifar10()

