# -*- coding:utf-8 -*-
# 
# Author: YIN MIAO
# Time: 2018/11/14 13:02

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def mnist():
    row, col = 4, 8
    data = sio.loadmat('./data/mnist_results.mat')['res']
    fig = plt.figure(figsize=(col, row))
    graph = gridspec.GridSpec(row, col)
    graph.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(data):
        ax = plt.subplot(graph[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.squeeze(sample), cmap='Greys_r')

    plt.savefig('mnist_tgan.png', bbox_inches='tight')
    plt.close(fig)



def cifar10():
    row, col = 4, 8
    data = sio.loadmat('./data/cifar10_results.mat')['res']
    fig = plt.figure(figsize=(col, row))
    graph = gridspec.GridSpec(row, col)
    graph.update(wspace=0.05, hspace=0.05)
    for i, sample in enumerate(data):
        ax = plt.subplot(graph[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(np.squeeze(sample), cmap='Greys_r')

    plt.savefig('cifar10_tgan.png', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    # cifar10()
    mnist()
