# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/8/26 21:01

import tensorflow as tf
import keras
import numpy as np
from skimage import transform
from aae import *
from matplotlib import pyplot as plt

batch_size = 32
epochs = 2000


if __name__ == '__main__':
    aaes = []
    aaes.append(AAE([7, 7, 1], [14, 14, 1], 32))
    aaes.append(AAE([14, 14, 1], [28, 28, 1], 64))

    (train_data, _), (_, _) = keras.datasets.mnist.load_data()
    train_data = (train_data.astype(np.float32) - 127.5) / 127.5

    datas = []
    data = transform.resize(train_data, [train_data.shape[0], 7, 7])
    data = np.expand_dims(data, axis=3)
    datas.append(data)

    data = transform.resize(train_data, [train_data.shape[0], 14, 14])
    data = np.expand_dims(data, axis=3)
    datas.append(data)

    data = np.expand_dims(train_data, axis=3)
    datas.append(data)

    for epoch in range(epochs):
        indices = np.random.randint(0, train_data.shape[0], batch_size)
        x = datas[0][indices]
        y = datas[1][indices]
        d_loss, g_loss, ae_loss = aaes[0].train(batch_size, x, y, 1)
        if epoch % 100 == 0:
            print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(epoch, d_loss, g_loss, ae_loss))
            aaes[0].save_samples(x, epoch, 'imgs1')

    for epoch in range(epochs):
        indices = np.random.randint(0, train_data.shape[0], batch_size)
        x = aaes[0].autoencoder.predict(datas[0][indices])
        y = datas[2][indices]
        d_loss, g_loss, ae_loss = aaes[1].train(batch_size, x, y, 1)
        if epoch % 100 == 0:
            print('epoch {}: D loss: {}, G loss: {}, AutoEncoder loss: {}'.format(epoch, d_loss, g_loss, ae_loss))
            aaes[1].save_samples(x, epoch, 'imgs2')

    index = np.random.randint(0, train_data.shape[0], 1)
    y1 = aaes[0].autoencoder.predict(datas[0][index])
    y2 = aaes[1].autoencoder.predict(y1)
    img1 = datas[1][index]
    img2 = datas[2][index]

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(221)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(img1), cmap='Greys_r')
    ax = fig.add_subplot(222)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(img2), cmap='Greys_r')
    ax = fig.add_subplot(223)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(y1), cmap='Greys_r')
    ax = fig.add_subplot(224)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    plt.imshow(np.squeeze(y2), cmap='Greys_r')

    # plt.savefig(file_name, bbox_inches='tight')
    plt.show()
    plt.close(fig)





