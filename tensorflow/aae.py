# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/22 17:54

import tensorflow as tf
import numpy as np
import sys
import scipy.io as sio
import os
import keras
from skimage import transform
from matplotlib import pyplot as plt


class AAE(object):

    def __init__(self, input_shape, output_shape, latent_dim):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape

        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.5)

        x = keras.Input(self.input_shape)

        self.discriminator = self._create_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.trainable = False

        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

        g = self.encoder(x)
        y = self.decoder(g)
        d = self.discriminator(g)

        self.autoencoder = keras.Model(x, y)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        self.generator = keras.Model(x, d)
        self.generator.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.summary()
        self.autoencoder.summary()

    def _create_encoder(self):
        x = keras.Input(shape=self.input_shape)
        h = keras.layers.Flatten()(x)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dense(512, activation='relu')(h)
        g = keras.layers.Dense(self.latent_dim)(h)

        return keras.Model(x, g)

    def _create_decoder(self):
        z = keras.Input(shape=(self.latent_dim,))
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dense(np.prod(self.output_shape), activation='tanh')(h)
        y = keras.layers.Reshape(self.output_shape)(h)

        return keras.Model(z, y)

    def _create_discriminator(self):
        z = keras.Input(shape=(self.latent_dim,))
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(256, activation='relu')(h)
        d = keras.layers.Dense(1, activation='sigmoid')(h)

        return keras.Model(z, d)

    def train(self, batch_size, train_data, iter_num, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)

        real = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

        for i in range(iter_num):
            indices = np.random.randint(0, train_data.shape[0], batch_size)
            xs = train_data[indices]

            latent_fake = self.encoder.predict(xs)
            latent_real = np.random.normal(size=[batch_size, self.latent_dim])

            d_loss_real = self.discriminator.train_on_batch(latent_real, real)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            g_loss = self.generator.train_on_batch(xs, real)

            ae_loss = self.autoencoder.train_on_batch(xs, xs)

            if i % 100 == 0:
                print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(i, d_loss, g_loss, ae_loss))
                samples = train_data[np.random.randint(0, train_data.shape[0], 8)]
                gen_samples = self.autoencoder.predict(samples)
                self.save_samples(samples, gen_samples, i, folder)

        return d_loss, g_loss, ae_loss

    @staticmethod
    def save_samples(samples, gen_samples, epoch, folder):

        fig = plt.figure(figsize=(samples.shape[0], 2))
        for i in range(samples.shape[0]):
            ax = fig.add_subplot(2, samples.shape[0], i+1)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.squeeze(samples[i]), cmap='Greys_r')

            ax = fig.add_subplot(2, samples.shape[0], samples.shape[0]+i+1)
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(np.squeeze(gen_samples[i]), cmap='Greys_r')

        if not os.path.exists('./{}/'.format(folder)):
            os.mkdir('./{}/'.format(folder))
        fig.savefig("./{}/iter_{}.png".format(folder, epoch), bbox_inches='tight')
        plt.close()

class SR(object):

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

        x = keras.Input(shape=self.input_shape)
        h = keras.layers.Flatten()(x)
        h = keras.layers.Dense(256, activation='relu')(h)
        h = keras.layers.Dense(512, activation='relu')(h)
        h = keras.layers.Dense(np.prod(self.output_shape), activation='tanh')(h)
        y = keras.layers.Reshape(self.output_shape)(h)
        self.srer = keras.Model(x, y)
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.5)
        self.srer.compile(optimizer=optimizer, loss='mse')

    def train(self, batch_size, x_data, y_data, iter_num):
        for i in range(iter_num):
            indices = np.random.randint(0, x_data.shape[0], batch_size)
            xs = x_data[indices]
            ys = y_data[indices]
            loss = self.srer.train_on_batch(xs, ys)
            if i % 100 == 0:
                print('SR loss={}'.format(loss))


if __name__ == '__main__':
    batch_size = 32
    epochs = 2000
    # (x_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    (x_train, _), (_, _) = keras.datasets.cifar10.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    # train_data = x_train[np.where(y_train == 1)[0]]
    # train_data = np.expand_dims(x_train, 3)
    train_data = x_train
    train_data0 = []
    for i in range(x_train.shape[0]):
        train_data0.append(transform.resize(x_train[i], (16, 16)))
    train_data0 = np.array(train_data0)
    # train_data0 = np.expand_dims(train_data0, 3)
    aae0 = AAE([16, 16, 3], [16, 16, 3], 128)
    aae0.train(batch_size, train_data0, epochs, 'img0')
    sr = SR([16, 16, 3], [32, 32, 3])
    sr.train(batch_size, train_data0, train_data, epochs)
    aae1 = AAE([32, 32, 3], [32, 32, 3], 256)
    aae1.train(batch_size, train_data, epochs, 'img1')

    indices = np.random.randint(0, train_data0.shape[0], 8)
    zs = np.random.normal(size=[batch_size, 128])
    samples = train_data0[indices]
    gs = aae0.decoder.predict(zs)
    gs = sr.srer.predict(gs)
    gs = aae1.autoencoder.predict(gs)
    AAE.save_samples(gs, gs, 0, 'img')




