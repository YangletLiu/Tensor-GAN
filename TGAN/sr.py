# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/10/10 22:54

import time
import os
import keras
import scipy.misc
import scipy.io as sio
import numpy as np
from init import *
from block_3d import *
from tensor_dl import *
from tensor_tsta import *
from tensor_product import *
from hyper_params import HyperParams as hp
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


def save_img(img, file_name):
    fig = plt.figure(figsize=(5, 15))
    ax = fig.add_subplot(111)
    plt.axis('off')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
    # plt.imshow(img, cmap='Greys_r')
    plt.imshow(img)
    plt.savefig(file_name, bbox_inches='tight')
    plt.close(fig)


def train_combined_dictionary(X_h, X_l, n_iter=5):
    X_p = np.concatenate([X_h, X_l], axis=0)
    print(X_h.shape)
    print(X_l.shape)
    print(X_p.shape)
    split_idx = X_h.shape[0]

    size_X_p = np.shape(X_p)
    X_p_hat = np.fft.fft(X_p, axis=-1)

    D = init_D(hp.patch_size, hp.r)
    C = np.zeros([hp.r, size_X_p[1], size_X_p[2]])

    if not os.path.exists('./out/'):
        os.mkdir('./out/')

    for i in range(n_iter):
        time_start = time.time()
        print('Iteration: {} / {}'.format(i+1, n_iter))

        C = tensor_tsta(X_p, D, C)
        D = tensor_dl(X_p_hat, C, hp.r)

        C = tensor_tsta(X_p, D, C)

        time_end = time.time()
        print('time:', time_end - time_start, 's')

    D_h = D[:split_idx]
    D_l = D[split_idx:]

    return D_h, D_l


def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


def extract_features(img):
    hf1 = np.array([[-1, 0, 1]])
    vf1 = hf1.T

    hf2 = np.array([[1, 0, -2, 0, 1]])
    vf2 = hf2.T

    def get_features(img, filter):
        features = np.zeros((img.shape))
        for i in range(img.shape[2]):
            features[:,:,i] = conv2(img[:,:,i], filter)
        return features

    fea_hf1 = get_features(img, hf1)
    fea_vf1 = get_features(img, vf1)
    fea_hf2 = get_features(img, hf2)
    fea_vf2 = get_features(img, vf2)

    return fea_hf1, fea_vf1, fea_hf2, fea_vf2


def sample_data(train_data, n_samples=10240):
    X_h = np.zeros((hp.patch_size*hp.patch_size, n_samples, hp.patch_size))
    X_l = np.zeros((hp.patch_size*hp.patch_size*4, n_samples, hp.patch_size))
    count = 0
    #蓄水池抽样
    for i in range(len(train_data)):
        img_h = train_data[i]
        img_l = scipy.misc.imresize(img_h, 0.5)
        img_l = scipy.misc.imresize(img_l, 200)

        img_l_fea1, img_l_fea2, img_l_fea3, img_l_fea4 = extract_features(img_l)

        img_l_fea1p = tensor_block_3d(img_l_fea1, hp.patch_size, hp.step)
        img_l_fea2p = tensor_block_3d(img_l_fea2, hp.patch_size, hp.step)
        img_l_fea3p = tensor_block_3d(img_l_fea3, hp.patch_size, hp.step)
        img_l_fea4p = tensor_block_3d(img_l_fea4, hp.patch_size, hp.step)

        img_hp = tensor_block_3d(img_h, hp.patch_size, hp.step)
        img_lp = np.concatenate([img_l_fea1p, img_l_fea2p, img_l_fea3p, img_l_fea4p], axis=0)
        cur_count = img_hp.shape[1]
        for j in range(cur_count):
            if count < n_samples:
                X_h[:,count,:] = img_hp[:,j,:]
                X_l[:,count,:] = img_lp[:,j,:]
            else:
                k = np.random.randint(0, count)
                if k < n_samples:
                    X_h[:,k,:] = img_hp[:,j,:]
                    X_l[:,k,:] = img_lp[:,j,:]
            count += 1

    idx = np.random.randint(0, len(x_train))
    img = x_train[idx]

    return X_h, X_l, img


def recover_img(D_h, D_l, img_l):
    img_l_fea1, img_l_fea2, img_l_fea3, img_l_fea4 = extract_features(img_l)

    img_l_fea1p = tensor_block_3d(img_l_fea1, hp.patch_size, hp.step)
    img_l_fea2p = tensor_block_3d(img_l_fea2, hp.patch_size, hp.step)
    img_l_fea3p = tensor_block_3d(img_l_fea3, hp.patch_size, hp.step)
    img_l_fea4p = tensor_block_3d(img_l_fea4, hp.patch_size, hp.step)

    img_lp = np.concatenate([img_l_fea1p, img_l_fea2p, img_l_fea3p, img_l_fea4p], axis=0)

    C0 = np.zeros((hp.r, img_lp.shape[1], img_lp.shape[2]))
    C = tensor_tsta(img_lp, D_l, C0)
    img_srp = tensor_product(D_h,'', C, '')
    img_sr = block_3d_tensor(img_srp, img_l.shape, hp.patch_size, hp.step)

    return img_sr


if __name__ == '__main__':
    (x_train, y_train), (_, _) = keras.datasets.cifar10.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    X_h, X_l, img_h = sample_data(train_data=x_train, n_samples=100)
    D_h, D_l = train_combined_dictionary(X_h, X_l)
    img_l = scipy.misc.imresize(img_h, 0.5)
    img_l = scipy.misc.imresize(img_l, 200)
    img_sr = recover_img(D_h, D_l, img_l)

    save_img(img_h, './out/h.png')
    save_img(img_l, './out/l.png')
    save_img(img_sr, './out/sr.png')

