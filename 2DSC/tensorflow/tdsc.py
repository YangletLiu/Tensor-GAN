# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/14 20:19

import tensorflow as tf
import numpy as np
import sys
import time
sys.path.append('../')

from matplotlib import pyplot as plt
from block_3d import *
from hyper_params import HyperParams as params


class Tdsc(object):

    def __init__(self, m, n, k):
        # size of X: m x n x k
        # size of D: m x r x k
        # size of C: r x n x k
        self.m = m
        self.n = n
        self.k = k
        # self.X_p = tf.placeholder(tf.float32, [batch_size, m, n, k])
        self.X_p = tf.placeholder(tf.float64, [m, n, k])
        self.D = tf.Variable(self.init_D(params.patch_size, params.r), dtype=tf.float64)
        self.C = tf.Variable(np.zeros([params.r, n, k]), dtype=tf.float64)
        self.dual_lambda = tf.Variable(np.random.rand(params.r), dtype=tf.float64)

        # tensor sparse coding
        self.C_assign = self.tensor_tsta(self.X_p, self.D, self.C)

        # tensor dictionry learning
        self.dl_loss, self.D_assign = self.tensor_dl(self.X_p, self.C, self.dual_lambda)
        self.dl_opt = tf.contrib.opt.ScipyOptimizerInterface(
            self.dl_loss, method='L-BFGS-B', var_to_bounds={self.dual_lambda:(0, np.infty)})

        # X reconstruction
        self.C_assign = self.tensor_tsta(self.X_p, self.D, self.C)
        self.X_p_recon = self.tensor_product(self.D, '', self.C, '')

    def tensor_dl(self, X_p, C, dual_lambda):
        X_p_hat = self.fft(tf.complex(X_p, tf.zeros(tf.shape(X_p), dtype=tf.float64)))
        C_hat = self.fft(tf.complex(C, tf.zeros(tf.shape(C), dtype=tf.float64)))
        x_hat_list = [tf.squeeze(x) for x in tf.split(X_p_hat, self.k, axis=-1)]
        c_hat_list = [tf.squeeze(c) for c in tf.split(C_hat, self.k, axis=-1)]

        cc_hat_list = [tf.matmul(c_hat, tf.transpose(c_hat)) for c_hat in c_hat_list]
        xc_hat_list = [tf.matmul(x_hat, tf.transpose(c_hat)) for (x_hat, c_hat)
                       in zip(x_hat_list, c_hat_list)]

        lambda_diag = tf.matrix_diag(dual_lambda)
        lambda_mat = tf.complex(lambda_diag, tf.zeros(tf.shape(lambda_diag), dtype=tf.float64))

        if self.m > params.r:
            f = sum([tf.trace(tf.matmul(self.pinv(cc_hat + lambda_mat),
                                        tf.matmul(tf.transpose(xc_hat), xc_hat)))
                    for (cc_hat, xc_hat) in zip(cc_hat_list, xc_hat_list)])
        else:
            f = sum([tf.trace(tf.matmul(tf.matmul(xc_hat, self.pinv(cc_hat + lambda_mat)),
                                        tf.transpose(xc_hat)))
                     for (cc_hat, xc_hat) in zip(cc_hat_list, xc_hat_list)])

        D_hat = tf.concat([tf.expand_dims(tf.transpose(tf.matmul(self.pinv(cc_hat + lambda_mat),
                                                                 tf.transpose(xc_hat))), axis=-1)
                           for (cc_hat, xc_hat) in zip(cc_hat_list, xc_hat_list)], axis=-1)

        D_ = tf.real(self.ifft(D_hat))
        D = tf.where(tf.is_nan(D_), tf.zeros_like(D_), D_)
        D_assign = tf.assign(self.D, D)
        min_obj = tf.real(f) + self.k * tf.reduce_sum(dual_lambda)
        return min_obj, D_assign

    def tensor_tsta(self, X_p, D, C):
        C0 = C

        DD = self.tensor_product(D, 't', D, '')
        # DD_cmat = blk_circ_mat(DD)
        # l0 = tf.norm(DD_cmat, 2)
        l0 = self.norm(DD)
        DX = self.tensor_product(D, 't', X_p, '')

        J1 = C0
        t1 = 1

        for i in range(params.tsta_max_iter):
            l1 = params.eta ** i * l0
            grad_J1 = self.tensor_product(DD, 't', J1, '') - DX
            temp = J1 - grad_J1 / l1  # tf.divide(x, y)
            C1 = tf.multiply(tf.sign(temp), tf.maximum(tf.abs(temp) - params.beta / l1, 0))
            t2 = (1 + np.sqrt(1 + 4 * t1 ** 2)) / 2
            J1 = C1 + tf.scalar_mul((t1 - 1) / t2, (C1 - C0))
            C0 = C1
            t1 = t2

        C_assign = tf.assign(self.C, C1)

        return C_assign

    def tensor_product(self, P, ch1, Q, ch2):
        P_hat = self.fft(tf.complex(P, tf.zeros(tf.shape(P), dtype=tf.float64)))
        Q_hat = self.fft(tf.complex(Q, tf.zeros(tf.shape(Q), dtype=tf.float64)))
        p_hat_list = [tf.squeeze(p) for p in tf.split(P_hat, self.k, axis=-1)]
        q_hat_list = [tf.squeeze(q) for q in tf.split(Q_hat, self.k, axis=-1)]

        if ch1 == 't' and ch2 == 't':
            S_hat = tf.concat([tf.expand_dims(tf.matmul(tf.transpose(p_hat), tf.transpose(q_hat)), axis=-1)
                               for (p_hat, q_hat) in zip(p_hat_list, q_hat_list)], axis=-1)
        elif ch1 == 't':
            S_hat = tf.concat([tf.expand_dims(tf.matmul(tf.transpose(p_hat), q_hat), axis=-1)
                               for (p_hat, q_hat) in zip(p_hat_list, q_hat_list)], axis=-1)
        elif ch2 == 't':
            S_hat = tf.concat([tf.expand_dims(tf.matmul(p_hat, tf.transpose(q_hat)), axis=-1)
                               for (p_hat, q_hat) in zip(p_hat_list, q_hat_list)], axis=-1)
        else:
            S_hat = tf.concat([tf.expand_dims(tf.matmul(p_hat, q_hat), axis=-1)
                               for (p_hat, q_hat) in zip(p_hat_list, q_hat_list)], axis=-1)

        return tf.real(self.ifft(S_hat))

    def train(self, sess, X_p, X, iter_num):
        for i in range(iter_num):
            time_start = time.time()
            print('Iteration: {} / {}'.format(i, params.sc_max_iter),)

            # compute tensor coefficients C
            sess.run(self.C_assign, feed_dict={self.X_p:X_p})

            # compute tensor dictionary D
            self.dl_opt.minimize(sess, feed_dict={self.X_p:X_p})
            sess.run(self.D_assign, feed_dict={self.X_p:X_p})

            # recover input tensor X
            sess.run(self.C_assign, feed_dict={self.X_p:X_p})
            X_p_recon = sess.run(self.X_p_recon)
            X_recon = block_3d_tensor(X_p_recon, np.shape(X))

            time_end = time.time()
            print('time:', time_end - time_start, 's')
        return X_recon

    @staticmethod
    def pinv_svd(a, rcond=1e-15):
        s, u, v = tf.svd(a)
        # Ignore singular values close to zero to prevent numerical overflow
        limit = rcond * tf.reduce_max(s)
        non_zero = tf.greater(s, limit)

        reciprocal = tf.where(non_zero, tf.reciprocal(s), tf.zeros(s.shape))
        lhs = tf.matmul(v, tf.complex(tf.matrix_diag(reciprocal), tf.zeros(tf.shape(tf.matrix_diag(reciprocal)))))
        return tf.matmul(lhs, u, transpose_b=True)

    @staticmethod
    def pinv(a):
        return tf.py_func(np.linalg.pinv, [a], tf.complex128)

    @staticmethod
    def fft(a):
        def np_fft(a):
            return np.fft.fft(a, axis=-1)
        return tf.py_func(np_fft, [a], tf.complex128)

    @staticmethod
    def ifft(a):
        def np_ifft(a):
            return np.fft.ifft(a, axis=-1)
        return tf.py_func(np_ifft, [a], tf.complex128)

    @staticmethod
    def norm(A):
        def blk_circ_mat_norm(A):
            sz_A = np.shape(A)
            dim = [0, 0]
            dim[0] = sz_A[0] * sz_A[2]
            dim[1] = sz_A[1] * sz_A[2]
            A_c = np.zeros(dim)
            A_mat = np.reshape(np.transpose(A, [0, 2, 1]), [dim[0], sz_A[1]], order='F')
            for k in range(sz_A[2]):
                A_c[:, k * sz_A[1]:(k + 1) * sz_A[1]] = np.roll(A_mat, k * sz_A[0], axis=0)
            return np.linalg.norm(A_c, 2)
        return tf.py_func(blk_circ_mat_norm, [A], tf.float64)

    @staticmethod
    def init_D(patch_size, r):
        D_mat = np.random.rand(patch_size ** 3, r) * 2 - 1
        D_mat_sum = np.sqrt(np.sum(np.square(D_mat), axis=0))
        for i in range(D_mat_sum.shape[0]):
            D_mat[:, i] /= D_mat_sum[i]
        D = np.transpose(np.reshape(
            D_mat, [patch_size ** 2, patch_size, params.r], order='F'), [0, 2, 1])
        return D

    @staticmethod
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

    Tdsc.save_img(X[:,:,2], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    tdsc = Tdsc(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tdsc.train(sess, X_p, X, params.sc_max_iter)

    sess.close()
