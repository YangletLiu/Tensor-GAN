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
        D_mat[:, i] /= D_mat_1[i]
    D = np.transpose(np.reshape( \
        D_mat, [patch_size ** 2, patch_size, params.r]), [0, 2, 1])
    return D


class TDSC(object):

    def __init__(self, n1, n2, n3, batch_size):
        self.X_p = tf.placeholder(tf.float32, [batch_size, n1, n2, n3])
        self.D = tf.Variable(init_D(params.patch_size, params.r), dtype=tf.float32)
        self.B = tf.Variable(np.zeros([params.r, n2, n3]), dtype=tf.float32)
        self.dual_lambda = tf.Variable(tf.float32, [params.r, 1])

        self.X_p_hat = tf.fft(tf.complex(self.X_p, tf.zeros([batch_size, n1, n2, n3])))

        self.dl_loss = self.compute_tensor_dl_loss(self.X_p_hat, self.B, self.dual_lambda)
        self.dl_opt = tf.contrib.opt.ScipyOptimizerInterface(self.dl_loss, method='L-BFGS-B')



    def compute_tensor_dl_loss(self, X_p_hat, B, dual_lambda):
        B_hat = tf.fft(tf.complex(B, tf.zeros([params.r, n2, n3])))
        m = n1
        k = n3
        x_hat_list = [tf.squeeze(t) for t in tf.split(X_p_hat, k)]
        b_hat_list = [tf.squeeze(t) for t in tf.split(B_hat, k)]

        BB_t = tf.concat([tf.expand_dims(tf.matmul(b_hat, tf.transepos(b_hat)), axis=-1) \
                          for b_hat in b_hat_list], axis=-1)

        XB_t = tf.concat([tf.expand_dims(tf.matmul(x_hat, tf.transepos(b_hat)), axis=-1) \
                          for (x_hat, b_hat) in zip(x_hat_list, b_hat_list)], axis=-1)

        bb_hat_list = [tf.matmul(b_hat, tf.transpose(b_hat)) for b_hat in b_hat_list]
        xb_hat_list = [tf.matmul(x_hat, tf.transpose(b_hat)) for (x_hat, b_hat)
                       in zip(x_hat_list, b_hat_list)]

        if m > params.r:
            f = sum([np.trace(tf.matmul(tf.matrix_inverse(bb_hat), tf.matmul(tf.transpose(xb_hat), xb_hat)))
                    for (bb_hat, xb_hat) in zip(bb_hat_list, xb_hat_list)])
        else:
            f = sum([np.trace(tf.matmul(tf.matmul(xb_hat, tf.matrix_inverse(bb_hat)), tf.transpose(xb_hat)))
                     for (bb_hat, xb_hat) in zip(bb_hat_list, xb_hat_list)])

        return tf.real(f)



        lambda_matrix = tf.diag(dual_lambda)
        tf.trace(tf.matmul())
