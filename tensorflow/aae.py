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

    def train(self, batch_size, x, latent_real, y, iter_num):

        real = np.ones([batch_size, 1])
        fake = np.zeros([batch_size, 1])

        for i in range(iter_num):

            latent_fake = self.encoder.predict(x)
            # latent_real = np.random.normal(size=[batch_size, self.latent_dim])

            d_loss_real = self.discriminator.train_on_batch(latent_real, real)
            d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            g_loss = self.generator.train_on_batch(x, real)

            ae_loss = self.autoencoder.train_on_batch(x, y)

        return d_loss, g_loss, ae_loss

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


if __name__ == '__main__':
    aae = AAE([32, 32, 3], [32, 32, 3], 256)
    batch_size = 32
    epochs = 10000
    (train_data, _), (_, _) = keras.datasets.cifar10.load_data()
    train_data = (train_data.astype(np.float32) - 127.5) / 127.5

    for epoch in range(epochs):
        indices = np.random.randint(0, train_data.shape[0], batch_size)
        latent_real = np.random.normal(size=[batch_size, aae.latent_dim])
        imgs = train_data[indices]
        d_loss, g_loss, ae_loss = aae.train(batch_size, imgs,latent_real, imgs, 1)
        if epoch % 100 == 0:
            print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(epoch, d_loss, g_loss, ae_loss))
            samples = train_data[np.random.randint(0, train_data.shape[0], 8)]
            gen_samples = aae.autoencoder.predict(samples)
            aae.save_samples(samples, gen_samples, epoch, 'imgs')




