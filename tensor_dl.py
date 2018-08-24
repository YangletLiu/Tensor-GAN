# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/2 22:11

import numpy as np
from hyper_params import HyperParams as params
from scipy.optimize import minimize


def tensor_dl(X_hat, C, r):
    C_hat = np.fft.fft(C, axis=-1)  # mabe incorrect
    dual_lambda = 10 * np.abs(np.random.rand(r, 1))
    m, _, k = np.shape(X_hat)

    CC_t = np.zeros([r, r, k], dtype=complex)
    XC_t = np.zeros([m, r, k], dtype=complex)

    for kk in range(k):
        x_hat_k = X_hat[:, :, kk]
        c_hat_k = C_hat[:, :, kk]

        CC_t[:, :, kk] = np.matmul(c_hat_k, c_hat_k.T)
        XC_t[:, :, kk] = np.matmul(x_hat_k, c_hat_k.T)

    bnds = tuple((0, np.infty) for _ in range(len(dual_lambda)))
    fun = lambda x: fobj_dict_dual(x, XC_t, CC_t, k)

    res = minimize(fun, dual_lambda, method='L-BFGS-B', bounds=bnds)

    LAMBDA = np.diag(res.x)
    D_hat = np.zeros([m, r, k], dtype=complex)

    for kk in range(k):
        CC_t_k = np.squeeze(CC_t[:, :, kk])
        XC_t_k = np.squeeze(XC_t[:, :, kk])
        D_hat_k_t = np.matmul(np.linalg.pinv(CC_t_k + LAMBDA), XC_t_k.T)
        D_hat[:, :, kk] = np.transpose(D_hat_k_t)

    D = np.fft.ifft(D_hat, axis=-1)
    D[np.where(np.isnan(D) == True)] = 0
    D = np.real(D)

    return D


def fobj_dict_dual(x, XC_t, CC_t, k):
    m = np.shape(XC_t)[0]
    r = len(x)
    LAMBDA = np.diag(x)

    f = 0
    # g = np.zeros([r, 1])
    # H = np.zeros([r, r])

    for kk in range(k):
        XC_t_k = XC_t[:, :, kk]
        CC_t_k = CC_t[:, :, kk]
        CC_t_inv = np.linalg.pinv(CC_t_k + LAMBDA)

        if m > r:
            f += np.trace(np.matmul(CC_t_inv, np.matmul(XC_t_k.T, XC_t_k)))
        else:
            f += np.trace(np.matmul(np.matmul(XC_t_k, CC_t_inv), XC_t_k.T))

        f = np.real(f + k * np.sum(x))

    return f


if __name__ == '__main__':
    X_hat = np.random.rand(4, 3, 2)
    C = np.random.rand(2, 3, 2)
    n_basis = 2
    D = tensor_dl(X_hat, C, n_basis)
    print(D)
