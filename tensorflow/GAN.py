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
        self.latent_dim = 100
        self.input_shape = [40, 40]
        self.output_shape = [80, 80]

        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.5)

        x = keras.Input(self.input_shape)

        self.discriminator = self._creat_discriminator()
        self.discriminator.compile(optimizer=optimizer, loss='binary_crossentropy')

        self.discriminator.trainable = False

        self.encoder = self._creat_encoder()
        self.encoder.compile(optimizer=optimizer, loss='binary_crossentripy')

        self.decoder = self._creat_decoder()

        g = self.encoder(x)
        y = self.decoder(g)
        d = self.discriminator(g)

        self.autoencoder = keras.Model(x, y)
        self.autoencoder.compile(optimizer=optimizer, loss='mse')

        self.generator = keras.Model(x, d)
        self.generator.compile(optimizer=optimizer, loss='binary_crossentripy')

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
        z = keras.Input(shape=self.latent_dim)
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(1024, activation='relu')(h)
        h = keras.layers.Dense(np.prod(self.input_shape), activation='tanh')(h)
        y = keras.layers.Reshape(self.output_shape)(h)

        return keras.Model(z, y)

    def _creat_discriminator(self):
        z = keras.Input(shape=self.latent_dim)
        h = keras.layers.Dense(512, activation='relu')(z)
        h = keras.layers.Dense(256, activation='lrelu')(h)
        d = keras.layers.Dense(1, activation='sigmoid')(h)

        return keras.Model(z, d)

    def train(self):
        (x_train, _), (_, _) = keras.datasets.mnist.load_data()

        x_train = (x_train.astype(np.float32)) - 127.5) / 127.5
        x_train = np.expand_dims(x_train, axis=3)

        real = np.ones(batch_size, 1)
        fake = np.ones(batch_size, 1)

        for epoch in range(epochs):

            latent_fake = self.encoder.predict(imgs)




