# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/26 21:01

import tensorflow as tf
import keras
import os
import scipy.io as sio

from skimage import transform
from aae import *
from matplotlib import pyplot as plt
from tdsc import Tdsc
from block_3d import *
from tensor_product import *
from hyper_params import HyperParams as params

from keras import backend as K
from keras.engine.topology import Layer
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
    stddev = 1. / tf.sqrt(input_dim / 2.)
    return tf.random_normal(shape=size, stddev=stddev)


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
            weight = tf.get_variable('weight', initializer=init_w)

            init_b = tf.zeros([output_dim])
            bias = tf.get_variable('bias', initializer=init_b)

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

        self.tdsc = Tdsc(input_shape[0], input_shape[1], input_shape[2])
        self.coeff_shape = (params.r, self.tdsc.n, self.tdsc.k)

        self.discriminator = Network()
        self.encoder = Network()

        self.z = tf.placeholder(tf.float64, (self.latent_dim,), name='z')

        with tf.variable_scope('generator'):
            self.g = self._create_encoder(self.z)
        self.y = self._create_decoder(self.g)

        with tf.variable_scope('discriminator') as scope:
            self.d_real = self._create_discriminator(self.tdsc.C)
            scope.reuse_variables()
            self.d_fake = self._create_discriminator(self.g)

        content_loss = tf.reduce_sum(tf.square(self.y - self.tdsc.X_p))

        disc_loss = -tf.reduce_mean(self.d_real) + tf.reduce_mean(self.d_fake)
        gen_loss = -tf.reduce_mean(self.d_fake)
        ae_loss = content_loss

        self.d_opt = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(disc_loss,
                   var_list=self.discriminator.weights+self.discriminator.biases)

        self.g_opt = tf.train.AdamOptimizer(
            learning_rate=0.001,
            beta1=0.5,
            beta2=0.9
        ).minimize(gen_loss,
                   var_list=self.discriminator.weights+self.discriminator.biases)



    def _create_encoder(self, z):
        h = self.encoder.dense(z, 256)
        h = lrelu(h)
        h = self.encoder.dense(h, 1024)
        h = lrelu(h)
        h = self.encoder.dense(h, np.prod(self.coeff_shape))
        h = tf.reshape(h, self.coeff_shape)

        return h

    def _create_decoder(self, coeff):
        y = self.tproduct(self.tdsc.D, coeff)
        return y

    def _create_discriminator(self, coeff):
        h = tf.layers.flatten(coeff)
        h = self.discriminator.dense(h, 1024)
        h = lrelu(h)
        h = self.discriminator.dense(h, 256)
        h = lrelu(h)
        h = self.discriminator.dense(h, 1)

        return h

    @staticmethod
    def tproduct(p, q):
        def product(p, q):
            tensor_product(p, '', q, '')
        return tf.py_func(product, [p, q], tf.float64)


    def train(self, batch_size, x, iter_num):

        real = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

        coeff_real = self.tdsc.C
        z = np.random.normal(size=(batch_size, self.latent_dim))

        for i in range(iter_num):
            coeff_fake = self.encoder.predict(z)
            d_loss_real = self.discriminator.train_on_batch(coeff_real, real)
            d_loss_fake = self.discriminator.train_on_batch(coeff_fake, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            g_loss = self.generator.train_on_batch(z, real)
            ae_loss = self.autoencoder.train_on_batch(z, x)
            if i % 10 == 0:
                print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(
                    i, d_loss, g_loss, ae_loss))


def train_tgan():
    X = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']
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

    for i in range(params.sc_max_iter):
        X_recon = tgan.tdsc.train(sess, X_p, X, 1)

    X_batch = np.expand_dims(X, axis=0)
    tgan.train(1, X_batch, 1000)


def tgan_aae():
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
    train_tgan()
