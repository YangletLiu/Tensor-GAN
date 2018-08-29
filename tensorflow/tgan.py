# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/26 21:01

import tensorflow as tf
import keras
import numpy as np
import os
import scipy.io as sio

from skimage import transform
from aae import *
from matplotlib import pyplot as plt
from tdsc import Tdsc
from block_3d import *
from hyper_params import HyperParams as params

batch_size = 1
epochs = 2000

def tgan():
    # (train_data, _), (_, _) = keras.datasets.cifar10.load_data()
    # train_data = (train_data.astype(np.float32) - 127.5) / 127.5
    # index = np.random.randint(0, train_data.shape[0])
    # print(index)
    # X = train_data[index]
    X = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']
    X_s = np.zeros([32, 32, 16])
    for i in range(16):
        X_s[:,:,i] = transform.resize(X[:,:,i], (32, 32))
    X = X_s

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    Tdsc.save_img(X[:,:,0], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    tdsc = Tdsc(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(params.sc_max_iter):
        X_recon = tdsc.train(sess, X_p, X, 1)

    Tdsc.save_img(X_recon[:,:,0], './out/recons.png')
    C = sess.run(tdsc.C)

    aae = AAE(np.shape(X), np.shape(X), np.prod(np.shape(C)))
    X = np.expand_dims(X, axis=0)
    latent_real = np.expand_dims(C.flatten(), axis=0)

    for i in range(epochs):
        d_loss, g_loss, ae_loss = aae.train(batch_size, X, latent_real, X, 1)
        print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(i, d_loss, g_loss, ae_loss))
        if i % 10 == 0:
            gen_samples = aae.autoencoder.predict(X)
            gen_samples = 0.5 * gen_samples + 0.5
            aae.save_samples(X[:,:,:,0], gen_samples[:,:,:,0], i, 'out')

    sess.close()


if __name__ == '__main__':
    tgan()
