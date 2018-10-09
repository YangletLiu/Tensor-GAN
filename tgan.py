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
from hyper_params import HyperParams as hp


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


def init_D(patch_size, r):
    D_mat = np.random.rand(patch_size ** 3, r) * 2 - 1
    D_mat_sum = np.sqrt(np.sum(np.square(D_mat), axis=0))
    for i in range(D_mat_sum.shape[0]):
        D_mat[:, i] /= D_mat_sum[i]
    D = np.transpose(np.reshape(
        D_mat, [patch_size ** 2, patch_size, hp.r], order='F'), [0, 2, 1])
    return D.astype(np.float32)


class Tgan(object):

    def __init__(self, tensor_shape, block_shape, r, z_dim):
        # size of X: m x n x k
        # size of D: m x r x k
        # size of C: r x n x k
        self.tensor_shape = tensor_shape
        self.block_shape = block_shape
        self.m = self.block_shape[0]
        self.n = self.block_shape[1]
        self.k = self.block_shape[2]
        self.r = r
        self.z_dim = z_dim
        self.latent_shape = (self.m, self.r, self.k)


        self.X_p = tf.placeholder(tf.float32, [hp.batch_size, self.m, self.n, self.k], name='X_p')
        self.z = tf.placeholder(tf.float32, [hp.batch_size, self.z_dim], name='z')
        self.gen_vars = []
        self.disc_vars = []

        with tf.variable_scope('generator'):
            self.Y_p = self._generator(self.z)

        with tf.variable_scope('discriminator') as scope:
            disc_real = self._discriminator(self.X_p)
            scope.reuse_variables()
            disc_fake = self._discriminator(self.Y_p)

        d_loss = -tf.reduce_mean(disc_real) + tf.reduce_mean(disc_fake)
        g_loss = -tf.reduce_mean(disc_fake)
        content_loss = tf.reduce_sum(tf.square(self.X_p - self.Y_p))

        alphas = tf.random_uniform(
            shape=[hp.batch_size, 1],
            minval=0.,
            maxval=1.
        )

        x_ = tf.reshape(self.X_p, [-1, self.m*self.n*self.k])
        g_ = tf.reshape(self.Y_p, [-1, self.m*self.n*self.k])

        differences = x_ - g_
        interpolates = x_ + alphas * differences
        interpolates = tf.reshape(interpolates, [-1, self.m, self.n, self.k])
        gradients = tf.gradients(self._discriminator(interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        self.disc_loss = hp.sigma2 * (d_loss + hp.beta * gradient_penalty)
        self.gen_loss = hp.sigma1 * content_loss + hp.sigma3 * g_loss

        self.disc_opt = tf.train.AdamOptimizer(
            learning_rate=params.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.disc_loss, var_list=self.disc_vars)
        self.gen_opt = tf.train.AdamOptimizer(
            learning_rate=hp.learning_rate,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.gen_loss, var_list=self.gen_vars)

    def _discriminator(self, img):
        layer_num = 0
        params = []

        h = tf.reshape(img, (-1, np.prod(self.block_shape)))
        h, p = self.dense(h, 1024, layer_num=layer_num)
        params.extend(p)
        layer_num += 1
        h = relu(h)

        h, p = self.dense(h, 256, layer_num=layer_num)
        params.extend(p)
        layer_num += 1
        h = relu(h)

        h, p = self.dense(h, 1, layer_num=layer_num)
        params.extend(p)
        layer_num += 1

        self.disc_vars = params

        return h

    def _generator(self, z):
        layer_num = 0
        params = []

        h, p = self.dense(z, 256, layer_num=layer_num)
        params.extend(p)
        layer_num += 1
        h = relu(h)

        # h, p = self.dense(h, 1024, layer_num=layer_num)
        # params.extend(p)
        # layer_num += 1
        # h = lrelu(h)

        h, p = self.dense(h, np.prod(self.latent_shape), layer_num=layer_num)
        params.extend(p)
        layer_num += 1
        h = relu(h)

        h = tf.reshape(h, [-1, self.latent_shape[0], self.latent_shape[1], self.latent_shape[2]])
        h, p = self.dc_product(h, layer_num=layer_num)
        params.extend(p)
        # h = tf.nn.sigmoid(h)
        layer_num += 1

        self.gen_vars = params
        return h

    def dc_product(self, input, layer_num):
        with tf.variable_scope('dc_product' + str(layer_num)):
            init_tw = tf.zeros((self.r, self.n, self.k))
            tweight = tf.get_variable('tweight', initializer=init_tw, dtype=tf.float32)
            # tbias = tf.get_variable('tbias', initializer=tf.zeros(self.block_shape), dtype=tf.float32)
            output = tf.concat([tf.expand_dims(self.tensor_product(tf.squeeze(t), tweight), axis=0)
                                for t in tf.split(input, hp.batch_size, axis=0)], axis=0)

            params = [tweight]

        return output, params

    def dense(self, input, output_dim, layer_num):
        with tf.variable_scope('dense' + str(layer_num)):
            input_dim = input.get_shape().as_list()[1]

            init_w = xavier_init([input_dim, output_dim])
            weight = tf.get_variable('weight', initializer=init_w, dtype=tf.float32)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b, dtype=tf.float32)

            output = tf.add(tf.matmul(input, weight), bias)
            params = [weight, bias]

        return output, params

    def train(self, sess, train_data, step_num):
        if not os.path.exists('./out/'):
            os.mkdir('./out/')
        if not os.path.exists('./backup/'):
            os.mkdir('./backup/')

        if tf.train.get_checkpoint_state('./backup/'):
            saver = tf.train.Saver()
            saver.restore(sess, './backup/')
            print('********Restore the latest trained parameters.********')

        for step in range(step_num):
            for _ in range(5):
                indices = np.random.randint(0, train_data.shape[0], size=(hp.batch_size,))
                zs = self.sample_z(hp.batch_size, self.z_dim)
                X_ps = train_data[indices]
                _, dl = sess.run([self.disc_opt, self.disc_loss], feed_dict={self.X_p:X_ps, self.z:zs})

            indices = np.random.randint(0, train_data.shape[0], (hp.batch_size,))
            zs = np.random.uniform(-1., 1., size=(hp.batch_size, self.z_dim))
            X_ps = train_data[indices]
            _, gl = sess.run([self.gen_opt, self.gen_loss], feed_dict={self.X_p:X_ps, self.z:zs})

            if step % 100 == 0:
                print('step:{}, disc_loss={}, gen_loss={}'.format(step, dl, gl))
                zs = self.sample_z(hp.batch_size, self.z_dim)
                Y_ps = sess.run(self.Y_p, feed_dict={self.z:zs})
                Y = block_3d_tensor(Y_ps[0], self.tensor_shape)
                self.save_img(Y, './out/{}.png'.format(str(step).zfill(8)))
                saver = tf.train.Saver()
                saver.save(sess, './backup/', write_meta_graph=False)

    @staticmethod
    def tensor_product(P, Q):
        P_hat = tf.fft(tf.complex(P, tf.zeros(tf.shape(P), dtype=tf.float32)))
        Q_hat = tf.fft(tf.complex(Q, tf.zeros(tf.shape(Q), dtype=tf.float32)))
        k1 = P.get_shape().as_list()[-1]
        k2 = Q.get_shape().as_list()[-1]
        assert k1 == k2, 'tensor product dim not match!'
        k = k1
        p_hat_list = [tf.squeeze(p) for p in tf.split(P_hat, k, axis=-1)]
        q_hat_list = [tf.squeeze(q) for q in tf.split(Q_hat, k, axis=-1)]

        S_hat = tf.concat([tf.expand_dims(tf.matmul(p_hat, q_hat), axis=-1)
                           for (p_hat, q_hat) in zip(p_hat_list, q_hat_list)], axis=-1)

        return tf.real(tf.ifft(S_hat))

    @staticmethod
    def sample_z(batch_size, z_dim):
        return np.random.uniform(-1., 1., (batch_size, z_dim))

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


def train_tgan():
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = x_train[np.where(y_train == 1)[0]]
    train_data = []
    for i in range(x_train.shape[0]):
        tmp = transform.resize(x_train[i], (16, 16))
        train_data.append(tensor_block_3d(tmp))
    train_data = np.array(train_data)

    tensor_shape = (16, 16, 3)
    block_shape = np.shape(train_data[0])

    tgan = Tgan(tensor_shape, block_shape, hp.r, hp.z_dim)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tgan.train(sess, train_data, 100000)

    sess.close()


if __name__ == '__main__':
    train_tgan()