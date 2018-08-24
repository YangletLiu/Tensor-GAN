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


class AAE(object):

    def __init__(self, input_shape, output_shape, latent_dim):
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.output_shape = output_shape

        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.5)

        x = keras.Input(self.input_shape)

        self.discriminator = self._creat_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.trainable = False

        self.encoder = self._creat_encoder()
        self.decoder = self._creat_decoder()

        g = self.encoder(x)
        y = self.decoder(g)
        d = self.discriminator(g)

        self.autoencoder = keras.Model(x, y)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        self.generator = keras.Model(x, d)
        self.generator.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.summary()
        self.autoencoder.summary()

    def _creat_encoder(self):
        x = keras.Input(shape=self.input_shape)
        h = keras.layers.Flatten()(x)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dense(512, activation='relu')(h)
        g = keras.layers.Dense(self.latent_dim, activation='sigmoid')(h)

        return keras.Model(x, g)

    def _creat_decoder(self):
        z = keras.Input(shape=(self.latent_dim,))
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dense(np.prod(self.output_shape), activation='tanh')(h)
        y = keras.layers.Reshape(self.output_shape)(h)

        return keras.Model(z, y)

    def _creat_discriminator(self):
        z = keras.Input(shape=(self.latent_dim,))
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(256, activation='relu')(h)
        d = keras.layers.Dense(1, activation='sigmoid')(h)

        return keras.Model(z, d)

    def train(self):
        batch_size = 1
        epochs = 10
        (x_train, _), (_, _) = keras.datasets.mnist.load_data()

        # x_train = (x_train.astype(np.float32)) - 127.5) / 127.5
        # x_train = np.expand_dims(x_train, axis=3)

        real = np.ones([batch_size, 1])
        fake = np.ones([batch_size, 1])

        sample = sio.loadmat('../samples/balloons_101_101_31.mat')['Omsi']

        for epoch in range(epochs):

            imgs = np.expand_dims(sample, axis=0)
            latent_fake = self.encoder.predict(imgs)
            latent_real = np.random.normal(size=[batch_size, self.latent_dim])

            d_loss_real = self.discriminator.train_on_batch(latent_real, real)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            g_loss = self.generator.train_on_batch(imgs, real)

            ae_loss = self.autoencoder.train_on_batch(imgs, imgs)

            print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(epoch, d_loss, g_loss, ae_loss))


if __name__ == '__main__':
    aae = AAE([101, 101, 31], [101, 101, 31], 256)
    aae.train()




