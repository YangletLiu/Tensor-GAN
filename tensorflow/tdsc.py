# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/14 20:19

import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import time

from block_3d import *
from hyper_params import HyperParams as params


class TDSC(object):

    def __init__(self, m, n, k, batch_size, D0):
        # size of X: m x n x k
        # size of D: m x r x k
        # size of B: r x n x k
        self.m = m
        self.n = n
        self.k = k
        self.batch_size = batch_size
        # self.X_p = tf.placeholder(tf.float32, [batch_size, m, n, k])
        self.X_p = tf.placeholder(tf.float64, [m, n, k])
        # self.D = tf.Variable(self.init_D(params.patch_size, params.r), dtype=tf.float64)
        self.D = tf.Variable(D0, dtype=tf.float64)
        self.B = tf.Variable(np.zeros([params.r, n, k]), dtype=tf.float64)
        self.dual_lambda = tf.Variable(np.random.rand(params.r), dtype=tf.float64)

        # self.X_p_complex = tf.complex(self.X_p, tf.zeros([m, n, k]))
        # self.D_complex = tf.complex(self.D, tf.zeros([m, params.r, k]))
        # self.B_complex = tf.complex(self.B, tf.zeros([params.r, n, k]))

        # tensor sparse coding
        self.B_assign = self.tensor_tsta(self.X_p, self.D, self.B)

        # tensor dictionry learning
        self.dl_loss, self.D_assign = self.tensor_dl(self.X_p, self.B, self.dual_lambda)
        self.dl_opt = tf.contrib.opt.ScipyOptimizerInterface(
            self.dl_loss, method='L-BFGS-B', var_to_bounds={self.dual_lambda:(0, np.infty)})

        # X reconstruction
        self.B_assign = self.tensor_tsta(self.X_p, self.D, self.B)
        self.X_p_recon = self.tensor_product(self.D, '', self.B, '')

    def tensor_dl(self, X_p, B, dual_lambda):
        X_p_hat = tf.fft(tf.complex(X_p, tf.zeros(tf.shape(X_p), dtype=tf.float64)))
        B_hat = tf.fft(tf.complex(B, tf.zeros(tf.shape(B), dtype=tf.float64)))
        x_hat_list = [tf.squeeze(x) for x in tf.split(X_p_hat, self.k, axis=-1)]
        b_hat_list = [tf.squeeze(b) for b in tf.split(B_hat, self.k, axis=-1)]

        # BB_t = tf.concat([tf.expand_dims(tf.matmul(b_hat, tf.transepos(b_hat)), axis=-1)
        #                   for b_hat in b_hat_list], axis=-1)
        #
        # XB_t = tf.concat([tf.expand_dims(tf.matmul(x_hat, tf.transepos(b_hat)), axis=-1)
        #                   for (x_hat, b_hat) in zip(x_hat_list, b_hat_list)], axis=-1)

        bb_hat_list = [tf.matmul(b_hat, tf.transpose(b_hat)) for b_hat in b_hat_list]
        xb_hat_list = [tf.matmul(x_hat, tf.transpose(b_hat)) for (x_hat, b_hat)
                       in zip(x_hat_list, b_hat_list)]

        lambda_diag = tf.matrix_diag(dual_lambda)
        lambda_mat = tf.complex(lambda_diag, tf.zeros(tf.shape(lambda_diag), dtype=tf.float64))

        if self.m > params.r:
            f = sum([tf.trace(tf.matmul(self.pinv(bb_hat + lambda_mat),
                                        tf.matmul(tf.transpose(xb_hat), xb_hat)))
                    for (bb_hat, xb_hat) in zip(bb_hat_list, xb_hat_list)])
        else:
            f = sum([tf.trace(tf.matmul(tf.matmul(xb_hat, self.pinv(bb_hat + lambda_mat)),
                                        tf.transpose(xb_hat)))
                     for (bb_hat, xb_hat) in zip(bb_hat_list, xb_hat_list)])

        D_hat = tf.concat([tf.expand_dims(tf.transpose(tf.matmul(self.pinv(bb_hat + lambda_mat),
                                                                 tf.transpose(xb_hat))), axis=-1)
                           for (bb_hat, xb_hat) in zip(bb_hat_list, xb_hat_list)], axis=-1)

        D_ = tf.real(tf.ifft(D_hat))
        D_assign = tf.assign(self.D, tf.where(tf.is_nan(D_), tf.zeros_like(D_), D_))
        return tf.real(f), D_assign

    def tensor_tsta(self, X_p, D, B):
        B0 = B

        DD = self.tensor_product(D, 't', D, '')
        # DD_cmat = blk_circ_mat(DD)
        # l0 = tf.norm(DD_cmat, 2)
        l0 = tf.norm(DD, 2)
        DX = self.tensor_product(D, 't', X_p, '')

        C1 = B0
        t1 = 1

        for i in range(params.tsta_max_iter):
            l1 = params.eta ** i * l0
            grad_C1 = self.tensor_product(DD, 't', C1, '') - DX
            temp = C1 - grad_C1 / l1  #tf.divide(x, y) ?
            B1 = tf.multiply(tf.sign(temp), tf.maximum(tf.abs(temp) - params.beta / l1, 0))
            t2 = (1 + np.sqrt(1 + 4 * t1 ** 2)) / 2
            C1 = B1 + tf.scalar_mul((t1 - 1) / t2, (B1 - B0))
            B0 = B1
            t1 = t2

        B_assign = tf.assign(self.B, B1)

        return B_assign

    def tensor_product(self, P, ch1, Q, ch2):
        P_hat = tf.fft(tf.complex(P, tf.zeros(tf.shape(P), dtype=tf.float64)))
        Q_hat = tf.fft(tf.complex(Q, tf.zeros(tf.shape(Q), dtype=tf.float64)))
        p_hat_list = [tf.squeeze(p) for p in tf.split(P_hat, self.k, axis=-1)]
        q_hat_list = [tf.squeeze(q) for q in tf.split(Q_hat, self.k, axis=-1)]

        # x_hat_list = [tf.squeeze(t) for t in tf.split(X_p_hat, k)]

        # XB_t = tf.concat([tf.expand_dims(tf.matmul(x_hat, tf.transepos(b_hat)), axis=-1)
        #                   for (x_hat, b_hat) in zip(x_hat_list, b_hat_list)], axis=-1)

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

        return tf.real(tf.ifft(S_hat))

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
    def init_D(patch_size, r):
        D_mat = np.random.rand(patch_size ** 3, r) * 2 - 1
        D_mat_1 = np.sqrt(np.sum(np.square(D_mat), axis=0))
        for i in range(D_mat_1.shape[0]):
            D_mat[:, i] /= D_mat_1[i]
        D = np.transpose(np.reshape(
            D_mat, [patch_size ** 2, patch_size, params.r]), [0, 2, 1])
        return D

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
    X = sio.loadmat('../samples/baloons_101_101_31.mat')['Omsi']
    D0 = sio.loadmat('../samples/D0.mat')['D0']

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    save_img(X[:,:,2], './out/origin.png')

    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)

    tdsc = TDSC(m, n, k, 1, D0)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        for i in range(params.sc_max_iter):
            time_s = time.time()
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

            time_e = time.time()
            print('time:', time_e - time_s, 's')

            plt.imshow(X_recon[:,:,1])
            plt.show()







