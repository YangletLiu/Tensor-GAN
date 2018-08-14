# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/14 20:19


import tensorflow as tf
import numpy as np
from hyper_params import HyperParams as params

def init_D(patch_size, r):
    D_mat = np.random.rand(patch_size ** 3, r) * 2 - 1
    D_mat_1 = np.sqrt(np.sum(np.square(D_mat), axis=0))
    for i in range(D_mat_1.shape[0]):
        D_mat[:,i] /= D_mat_1[i]
    D = np.transpose(np.reshape( \
        D_mat, [patch_size ** 2, patch_size, params.r]), [0, 2, 1])
    return D

class TDSC(object):

    def __init__(self, n1, n2, n3, batch_size):
        self.X_p = tf.placeholder(tf.float32, [batch_size, n1, n2, n3])
        self.dual_lambda = tf.placeholder(tf.float32, [params.r, 1])
        self.D = tf.Variable(init_D(params.patch_size, params.r), dtype=tf.float32)
        self.B = tf.Variable(np.zeros([params.r, n2, n3]), dtype=tf.float32)

        self.X_p_hat = tf.fft(tf.complex(self.X_p, tf.zeros([batch_size, n1, n2, n3])))


    def tensor_dl(self, X_p_hat, D, B):







