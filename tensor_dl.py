# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/2 22:11

import numpy as np
from hyper_params import HyperParams as params
from scipy.optimize import minimize


def tensor_dl(X_hat, B, r):
    B_hat = np.fft.fft(B, axis=-1)  # mabe incorrect
    dual_lambda = 10 * np.abs(np.random.rand(r, 1))
    m, _, k = np.shape(X_hat)

    BB_t = np.zeros([r, r, k])
    XB_t = np.zeros([m, r, k])

    for kk in range(k):
        x_hat_k = X_hat[:, :, kk]
        b_hat_k = B_hat[:, :, kk]

        BB_t[:, :, kk] = np.matmul(b_hat_k, b_hat_k.T)
        XB_t[:, :, kk] = np.matmul(x_hat_k, b_hat_k.T)

    bnds = tuple([0, None] for _ in range(len(dual_lambda)))
    fun = lambda x: fobj_dict_dual(x, XB_t, BB_t, k)

    res = minimize(fun, dual_lambda, method='Nelder-Mead', bounds=bnds)

    LAMBDA = np.diag(res.x)
    D_hat = np.zeros([m, r, k])

    for kk in range(k):
        BB_t_k = np.squeeze(BB_t[:, :, kk])
        XB_t_k = np.squeeze(XB_t[:, :, kk])
        D_hat_k_t = np.matmul(np.linalg.pinv(BB_t_k + LAMBDA), XB_t_k.T)
        D_hat[:, :, kk] = np.transpose(D_hat_k_t)

    D = np.fft.ifft(D_hat, axis=-1)
    D[np.where(D == None)] = 0
    D = np.real(D)

    return D

def fobj_dict_dual(x, XB_t, BB_t, k):
    m = np.shape(XB_t)[1]
    r = len(x)
    LAMBDA = np.diag(x)

    f = 0
    # g = np.zeros([r, 1])
    # H = np.zeros([r, r])

    for kk in range(k):
        XB_t_k = XB_t[:, :, kk]
        BB_t_k = BB_t[:, :, kk]
        BB_t_inv = np.linalg.pinv(BB_t_k + LAMBDA)

        if m > r:
            f += np.trace(np.matmul(BB_t_inv, np.matmul(XB_t_k.T, XB_t_k)))
        else:
            f += np.trace(np.matmul(np.matmul(XB_t_k, BB_t_inv), XB_t_k.T))

        f = np.real(f + k * np.sum(x))

    return f


if __name__ == '__main__':
    X_hat = np.random.rand(4, 3, 2)
    B = np.random.rand(2, 3, 2)
    n_basis = 2
    D = tensor_dl(X_hat, B, n_basis)
    print(D)
