# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/28 16:54


import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys
import time
import os
import keras
from skimage import transform

from matplotlib import pyplot as plt
from block_3d import *
from hyper_params import HyperParams as params

ALPHA = 0.01
LAMBDA = 10
SIGMA = 1e-3


def lrelu(x, alpha=0.2):
    return tf.maximum(alpha*x, x)


def relu(x):
    return tf.nn.relu(x)


def elu(x):
    return tf.nn.elu(x)


def xavier_init(size):
    input_dim = size[0]
    stddev = 1. / np.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev, dtype=tf.float32)


def he_init(size, stride):
    input_dim = size[2]
    output_dim = size[3]
    filter_size = size[0]

    fan_in = input_dim * filter_size**2
    fan_out = output_dim * filter_size**2 / (stride**2)
    stddev = tf.sqrt(4. / (fan_in + fan_out))
    minval = -stddev * np.sqrt(3)
    maxval = stddev * np.sqrt(3)
    return tf.random_uniform(shape=size, minval=minval, maxval=maxval)

class Network(object):
    def __init__(self):
        self.layer_num = 0
        self.weights = []
        self.biases = []

    def conv2d(self, input, input_dim, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('conv'+str(self.layer_num)):

            init_w = he_init([filter_size, filter_size, input_dim, output_dim], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d(
                input,
                weight,
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def deconv2d(self, input, output_dim, filter_size, stride, padding='SAME'):
        with tf.variable_scope('deconv'+str(self.layer_num)):
            input_shape = input.get_shape().as_list()
            init_w = he_init([filter_size, filter_size, output_dim, input_shape[3]], stride)
            weight = tf.get_variable(
                'weight',
                initializer=init_w
            )

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable(
                'bias',
                initializer=init_b
            )

            output = tf.add(tf.nn.conv2d_transpose(
                value=input,
                filter=weight,
                output_shape=[
                    tf.shape(input)[0],
                    input_shape[1]*stride,
                    input_shape[2]*stride,
                    output_dim
                ],
                strides=[1, stride, stride, 1],
                padding=padding
            ), bias)
            output = tf.reshape(output, [tf.shape(input)[0], input_shape[1]*stride, input_shape[2]*stride, output_dim])

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output

    def batch_norm(self, input, scale=False):
        ''' batch normalization
        ArXiv 1502.03167v3 '''
        with tf.variable_scope('batch_norm'+str(self.layer_num)):
            output = tf.contrib.layers.batch_norm(input, scale=scale)
            self.layer_num += 1

        return output

    def dense(self, input, output_dim):
        with tf.variable_scope('dense'+str(self.layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w, dtype=tf.float32)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b, dtype=tf.float32)

            output = tf.add(tf.matmul(input, weight), bias)

            self.layer_num += 1
            self.weights.append(weight)
            self.biases.append(bias)

        return output


class Atsc(object):

    def __init__(self, m, n, k):
        # size of X: m x n x k
        # size of D: m x r x k
        # size of C: r x n x k
        self.m = m
        self.n = n
        self.k = k
        # self.X_p = tf.placeholder(tf.float32, [batch_size, m, n, k])
        self.X_p = tf.placeholder(tf.float32, [m, n, k], name='X_p')
        self.D = tf.Variable(self.init_D(params.patch_size, params.r), name='D', dtype=tf.float32)
        self.C = tf.Variable(np.zeros([params.r, n, k]), name='C', dtype=tf.float32)
        self.dual_lambda = tf.Variable(np.random.rand(params.r), name='lambda', dtype=tf.float32)
        self.disc_vars = []

        # tensor coefficients learning
        self.C_tsta = self.tensor_tsta(self.X_p, self.D, self.C)
        content_loss, self.X_p_recon = self.tensor_cl(self.X_p, self.D, self.C)

        # tensor dictionary learning
        self.dl_loss, self.D_assign = self.tensor_dl(self.X_p, self.C, self.dual_lambda)
        self.dl_opt = tf.contrib.opt.ScipyOptimizerInterface(
            self.dl_loss, method='L-BFGS-B', var_to_bounds={self.dual_lambda:(0, np.infty)})

        with tf.variable_scope('discriminator') as scope:
            disc_real = self._discriminator(self.X_p)
            scope.reuse_variables()
            disc_fake = self._discriminator(self.X_p_recon)

        d_loss = -tf.reduce_mean(disc_real) + tf.reduce_mean(disc_fake)
        g_loss = -tf.reduce_mean(disc_fake)
        beta = tf.random_uniform(
            shape=(1,),
            minval=0.,
            maxval=1.
        )

        differences = self.X_p - self.X_p_recon
        interpolates = self.X_p + beta * differences
        gradients = tf.gradients(self._discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean(tf.square(slopes - 1.))

        self.disc_loss = SIGMA * (d_loss + LAMBDA * gradient_penalty)
        self.disc_opt = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.disc_loss, var_list=self.disc_vars)
        self.cl_loss = content_loss + SIGMA * g_loss
        self.cl_opt = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.cl_loss, var_list=[self.C])


    def tensor_dl(self, X_p, C, dual_lambda):
        X_p_hat = self.fft(tf.complex(X_p, tf.zeros(tf.shape(X_p), dtype=tf.float32)))
        C_hat = self.fft(tf.complex(C, tf.zeros(tf.shape(C), dtype=tf.float32)))
        x_hat_list = [tf.squeeze(x) for x in tf.split(X_p_hat, self.k, axis=-1)]
        c_hat_list = [tf.squeeze(c) for c in tf.split(C_hat, self.k, axis=-1)]

        cc_hat_list = [tf.matmul(c_hat, tf.transpose(c_hat)) for c_hat in c_hat_list]
        xc_hat_list = [tf.matmul(x_hat, tf.transpose(c_hat)) for (x_hat, c_hat)
                       in zip(x_hat_list, c_hat_list)]

        lambda_diag = tf.matrix_diag(dual_lambda)
        lambda_mat = tf.complex(lambda_diag, tf.zeros(tf.shape(lambda_diag), dtype=tf.float32))

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

        C_tsta = tf.assign(self.C, C1)
        return C_tsta

    def tensor_cl(self, X_p, D, C):
        X_p_recon = self.tensor_product(D, '', C, '')
        content_loss = tf.reduce_sum(tf.square(X_p_recon - X_p)) + ALPHA * tf.reduce_sum(tf.abs(C))

        return content_loss, X_p_recon

    def _discriminator(self, img):
        discriminator = Network()
        h = tf.reshape(img, (1, np.prod((self.m, self.n, self.k))))
        h = discriminator.dense(h, 1024)
        h = lrelu(h)
        h = discriminator.dense(h, 256)
        h = lrelu(h)
        h = discriminator.dense(h, 1)
        self.disc_vars = discriminator.weights + discriminator.biases

        return h

    def tensor_product(self, P, ch1, Q, ch2):
        P_hat = tf.fft(tf.complex(P, tf.zeros(tf.shape(P), dtype=tf.float32)))
        Q_hat = tf.fft(tf.complex(Q, tf.zeros(tf.shape(Q), dtype=tf.float32)))
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

        return tf.real(tf.ifft(S_hat))

    def train(self, sess, X, X_p, iter_num):

        for i in range(params.sc_max_iter):
            sess.run(self.C_tsta, feed_dict={self.X_p:X_p})
            self.dl_opt.minimize(sess, feed_dict={self.X_p:X_p})
            sess.run(self.D_assign, feed_dict={self.X_p:X_p})
            sess.run(self.C_tsta, feed_dict={self.X_p:X_p})
            X_p_recon = sess.run(self.X_p_recon)
            X_recon = block_3d_tensor(X_p_recon, np.shape(X))

        # self.save_img(X_recon, './out/atsc_init.png')

        for i in range(iter_num):
            # time_start = time.time()
            # print('Iteration: {} / {}'.format(i+1, iter_num))

            for _ in range(5):
                sess.run(self.disc_opt, feed_dict={self.X_p:X_p})

            # compute tensor coefficients C
            for j in range(1):
                _, loss = sess.run([self.cl_opt, self.cl_loss], feed_dict={self.X_p:X_p})

            # compute tensor dictionary D
            self.dl_opt.minimize(sess, feed_dict={self.X_p:X_p})
            sess.run(self.D_assign, feed_dict={self.X_p:X_p})

            # recover input tensor X
            for j in range(1):
                _, loss = sess.run([self.cl_opt, self.cl_loss], feed_dict={self.X_p:X_p})

            # print(loss)
            if i % 10 == 0:
                X_p_recon = sess.run(self.X_p_recon)
                X_recon = block_3d_tensor(X_p_recon, np.shape(X))

                # self.save_img(X_recon, './out/atsc_{}.png'.format(str(i).zfill(2)))

            # time_end = time.time()
            # print('time:', time_end - time_start, 's')
        return loss, X_recon

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
        def np_pinv(a):
            return np.linalg.pinv(a).astype(np.complex64)
        return tf.py_func(np_pinv, [a], tf.complex64)

    @staticmethod
    def fft(a):
        def np_fft(a):
            return np.fft.fft(a, axis=-1).astype(np.complex64)
        return tf.py_func(np_fft, [a], tf.complex64)

    @staticmethod
    def ifft(a):
        def np_ifft(a):
            return np.fft.ifft(a, axis=-1).astype(np.complex64)
        return tf.py_func(np_ifft, [a], tf.complex64)

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
            return np.linalg.norm(A_c, 2).astype(np.float32)
        return tf.py_func(blk_circ_mat_norm, [A], tf.float32)

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
        img = 0.5 * img + 0.5
        fig = plt.figure(figsize=(5, 15))
        ax = fig.add_subplot(111)
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img, cmap='Greys_r')
        plt.savefig(file_name, bbox_inches='tight')
        plt.close(fig)


def train_atsc():
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    train_data = []
    for x, y in zip(x_train, y_train):
        if y == 1:
            train_data.append((x.astype(np.float32) - 127.5) / 127.5)

    # train_data = (train_data.astype(np.float32) - 127.5) / 127.5
    # X = sio.loadmat('./samples/balloons_101_101_31.mat')['Omsi']
    # X_s = np.zeros([32, 32, 16])
    # for i in range(16):
    #     X_s[:,:,i] = transform.resize(X[:,:,i], (32, 32))
    # X = X_s
    if not os.path.exists('./out/'):
        os.mkdir('./out/')
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./backup/latest/'):
        os.mkdir('./backup/latest/')
    # Atsc.save_img(X, './out/origin.png')

    index = np.random.randint(0, len(train_data))
    X = train_data[index]
    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    atsc = Atsc(m, n, k)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    if tf.train.get_checkpoint_state('./backup/latest/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest/')
        print('********Restore the latest trained parameters.********')

    for step in range(100000):

        index = np.random.randint(0, len(train_data))
        X = train_data[index]
        X_p = tensor_block_3d(X)

        loss, X_recon = atsc.train(sess, X, X_p, 1)
        if step % 100 == 0:
            saver = tf.train.Saver()
            saver.save(sess, './backup/latest/', write_meta_graph=False)
            print('step:{}, loss:{}'.format(step, loss))
            Atsc.save_img(X_recon, './out/atsc_{}.png'.format(str(step).zfill(8)))

    sess.close()

def eval_atsc():
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    train_data = []
    for x, y in zip(x_train, y_train):
        if y == 1:
            train_data.append((x.astype(np.float32) - 127.5) / 127.5)

    # train_data = (train_data.astype(np.float32) - 127.5) / 127.5
    # X = sio.loadmat('./samples/balloons_101_101_31.mat')['Omsi']
    # X_s = np.zeros([32, 32, 16])
    # for i in range(16):
    #     X_s[:,:,i] = transform.resize(X[:,:,i], (32, 32))
    # X = X_s
    if not os.path.exists('./eval/'):
        os.mkdir('./eval/')
    if not os.path.exists('./backup/'):
        os.mkdir('./backup/')
    if not os.path.exists('./backup/latest/'):
        os.mkdir('./backup/latest/')

    index = np.random.randint(0, len(train_data))
    X = train_data[index]
    X_p = tensor_block_3d(X)
    m, n, k = np.shape(X_p)
    atsc = Atsc(m, n, k)
    C_input = tf.placeholder(tf.float32, [params.r, n, k])
    C_assign = tf.assign(atsc.C, C_input)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    if tf.train.get_checkpoint_state('./backup/latest/'):
        saver = tf.train.Saver()
        saver.restore(sess, './backup/latest/')
        print('********Restore the latest trained parameters.********')

    index = np.random.randint(0, len(train_data))
    X = train_data[index]
    X_p = tensor_block_3d(X)
    Atsc.save_img(X, 'eval/img1.png')
    C1 = sess.run(atsc.C_tsta, feed_dict={atsc.X_p:X_p})
    C1 = sess.run(atsc.C)

    index = np.random.randint(0, len(train_data))
    X = train_data[index]
    X_p = tensor_block_3d(X)
    Atsc.save_img(X, 'eval/img2.png')
    C2 = sess.run(atsc.C_tsta, feed_dict={atsc.X_p:X_p})
    C2 = sess.run(atsc.C)

    C3 = (C1)

    sess.run(C_assign, feed_dict={C_input:C3})

    X_p_recon = sess.run(atsc.X_p_recon)
    X_recon = block_3d_tensor(X_p_recon, np.shape(X))

    Atsc.save_img(X_recon, 'eval/recon.png')

    sess.close()

if __name__ == '__main__':
    train_atsc()
    # eval_atsc()


