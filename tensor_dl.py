# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/2 22:11

import numpy as np


def tensor_dl(X_hat, S, n_basis):
    S_hat = np.fft.fftn(S, axes=3) #mabe incorrect
    dual_lambda = 10 * np.abs(np.random.rand(n_basis, 1))
    m, _, k = np.shape(X_hat)
    r = n_basis

    SS_t = np.zeros([r, r, k])
    XS_t = np.zeros([m, r, k])

    for kk in range(k)
        x_hat_k = X_hat[:,:,kk]
        s_hat_k = S_hat[:,:,kk]

        SS_t[:,:,kk] = s_hat_k * np.transpose(s_hat_k)
        XS_t[:,:,kk] = x_hat_k * np.transpose(x_hat_k)


    lb = np.zeros(np.shape(dual_lambda))

    #x = optimize xxxxxx

    LAMBDA = np.diag(x)
    B_hat = np.zeros([m, r, k])

    for kk in range(k):
        SS_t_k = np.squeeze(SS_t[:,:,kk])
        XS_t_k = np.squeeze(XS_t[:,:,kk])
        B_hat_k_t = np.linalg.pinv(SS_t_k + LAMBDA) * XS_t_k
        B_hat[:,:,kk] = np.transpose(B_hat_k_t)

    B = np.fft.ifftn(B_hat, axes=3)
    B[np.where(B==None)] = 0
    B = np.real(B)

    return B

def fobj_basis_dual(lmd, XS_t)

    m = np.shape(XS_t)[1]
    r = len(lmd)
    LAMBDA = np.diag(lmd)

    f = 0
    #g = np.zeros([r, 1])
    #H = np.zeros([r, r])

    for kk in range(k):
        XS_t_k = XS_t[:,:,kk]
        SS_t_k = SS_t[:,:,kk]
        SS_t_inv = np.linalg.pinv(SS_t_k + LAMBDA)

        if m > r:
            f += np.trace(SS_t_inv * (np.transpose(XS_t_k) * XS_t_k))
        else:
            f += np.trace(XS_t_k * SS_t_inv * np.transpose(XS_t_k))

        f = np.real(f + k * np.sum(lmd))

        return f
