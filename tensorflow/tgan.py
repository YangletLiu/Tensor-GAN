# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/26 21:01

import tensorflow as tf
import os
import scipy.io as sio

from skimage import transform
from matplotlib import pyplot as plt
from tdsc import Tdsc
from block_3d import *
from hyper_params import HyperParams as params

import numpy as np

batch_size = 1
epochs = 2000


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

class Tgan(object):
    def __init__(self, input_shape, output_shape, latent_dim):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.m = input_shape[0]
        self.n = input_shape[1]
        self.k = input_shape[2]

        self.tdsc = Tdsc(input_shape[0], input_shape[1], input_shape[2])
        self.coeff_shape = (params.r, self.tdsc.n, self.tdsc.k)

        self.z = tf.placeholder(tf.float32, (1, self.latent_dim), name='z')
        self.x = tf.placeholder(tf.float32, self.input_shape, name='x')

        self.d_vars = []
        self.enc_vars = []

        with tf.variable_scope('generator'):
            self.g = self._create_encoder(self.z)
            self.y = self._create_decoder(self.g)

        with tf.variable_scope('discriminator') as scope:
            self.d_real = self._create_discriminator(self.tdsc.C)
            scope.reuse_variables()
            self.d_fake = self._create_discriminator(self.g)

        content_loss = tf.reduce_sum(tf.square(self.y - self.x))

        # self.d_loss = -tf.reduce_mean(self.d_real) + tf.reduce_mean(self.d_fake)
        # self.g_loss = -tf.reduce_mean(self.d_fake)
        self.ae_loss = content_loss
        real = tf.constant([[1]], dtype=tf.float32)
        fake = tf.constant([[0]], dtype=tf.float32)
        self.d_loss_real = tf.nn.sigmoid_cross_entropy_with_logits(labels=real, logits=self.d_real)
        self.d_loss_fake = tf.nn.sigmoid_cross_entropy_with_logits(labels=fake, logits=self.d_fake)
        self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=real, logits=self.d_fake)

        self.d_opt_real = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.d_loss_real,
                   var_list=self.d_vars)

        self.d_opt_fake = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.d_loss_fake,
                   var_list=self.d_vars)

        self.g_opt = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.g_loss,
                   var_list=self.enc_vars)

        # self.d_opt = tf.train.AdamOptimizer(
        #     learning_rate=0.001,
        #     beta1=0.5,
        #     beta2=0.9
        # ).minimize(self.d_loss,
        #            var_list=self.d_vars)
        #
        # self.g_opt = tf.train.AdamOptimizer(
        #     learning_rate=0.001,
        #     beta1=0.5,
        #     beta2=0.9
        # ).minimize(self.g_loss,
        #            var_list=self.enc_vars)

        self.ae_opt = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(self.ae_loss,
                   var_list=self.enc_vars)

    def _create_encoder(self, z):
        encoder = Network()
        h = encoder.dense(z, 256)
        h = lrelu(h)
        h = encoder.dense(h, 1024)
        h = lrelu(h)
        h = encoder.dense(h, np.prod(self.coeff_shape))
        h = tf.reshape(h, self.coeff_shape)
        self.enc_vars = encoder.weights + encoder.biases

        return h

    def _create_decoder(self, coeff):
        y = self.tensor_product(self.tdsc.D, '', coeff, '')
        return y

    def _create_discriminator(self, coeff):
        discriminator = Network()
        # h = tf.layers.flatten(coeff)
        h = tf.reshape(coeff, (1, np.prod(self.coeff_shape)))
        h = discriminator.dense(h, 1024)
        h = lrelu(h)
        h = discriminator.dense(h, 256)
        h = lrelu(h)
        h = discriminator.dense(h, 1)
        self.d_vars = discriminator.weights + discriminator.biases

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
            X_recon = self.tdsc.train(sess, X, X_p, 1)
            Tdsc.save_img(X_recon[:,:,0], './out/{}.png'.format(i))

        zs = np.random.normal(size=(1, self.latent_dim))

        for i in range(iter_num):
            _, d_loss_real = sess.run([self.d_opt_real, self.d_loss_real], feed_dict={self.z:zs})
            _, d_loss_fake = sess.run([self.d_opt_fake, self.d_loss_fake], feed_dict={self.z:zs})
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            _, g_loss = sess.run([self.g_opt, self.g_loss], feed_dict={self.z:zs})
            _, ae_loss = sess.run([self.ae_opt, self.ae_loss], feed_dict={self.z:zs, self.x:X_p})
            if i % 10 == 0:
                print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(
                    i, d_loss, g_loss, ae_loss))
                y = sess.run(self.y, feed_dict={self.z:zs})
                X_recon = block_3d_tensor(y, np.shape(X))
                X_recon = np.expand_dims(X_recon, axis=0)
                X_batch = np.expand_dims(X, axis=0)
                self.save_samples(X_batch[:,:,:,0], X_recon[:,:,:,0], i, 'out')

    def save_samples(self, samples, gen_samples, epoch, folder):

        fig = plt.figure(figsize=(samples.shape[0], 2))
        for i in range(samples.shape[0]):
            ax = fig.add_subplot(2, samples.shape[0], 2*i+1)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.squeeze(samples[i]), cmap='Greys_r')

            ax = fig.add_subplot(2, samples.shape[0], 2*i+2)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.squeeze(gen_samples[i]), cmap='Greys_r')

        if not os.path.exists('./{}/'.format(folder)):
            os.mkdir('./{}/'.format(folder))
        fig.savefig("./{}/iter_{}.png".format(folder, epoch), bbox_inches='tight')
        plt.close()

    @staticmethod
    def fft(a):
        def np_fft(a):
            return np.fft.fft(a, axis=-1)
        return tf.py_func(np_fft, [a], tf.complex64)

    @staticmethod
    def ifft(a):
        def np_ifft(a):
            return np.fft.ifft(a, axis=-1)
        return tf.py_func(np_ifft, [a], tf.complex64)

def train_tgan():
    X = sio.loadmat('./samples/balloons_101_101_31.mat')['Omsi']
    X_s = np.zeros([32, 32, 16])
    for i in range(16):
        X_s[:,:,i] = transform.resize(X[:,:,i], (32, 32))
    X = X_s

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    Tdsc.save_img(X[:,:,0], './out/origin.png')
    X_p = tensor_block_3d(X)

    tgan = Tgan(np.shape(X_p), np.shape(X_p), 128)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    tgan.train(sess, X, X_p, 10000)
    sess.close()


def tgan_aae():
    # (train_data, _), (_, _) = keras.datasets.cifar10.load_data()
    # train_data = (train_data.astype(np.float32) - 127.5) / 127.5
    # index = np.random.randint(0, train_data.shape[0])
    # print(index)
    # X = train_data[index]
    X = sio.loadmat('./samples/balloons_101_101_31.mat')['Omsi']
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

    Tdsc.save_img(X_recon, './out/recons.png')
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
    train_tgan()
