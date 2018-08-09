# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/2 22:11

import numpy as np
from scipy.optimize import minimize


def tensor_dl(X_hat, S, n_basis):
    S_hat = np.fft.fft(S, axis=-1)  # mabe incorrect
    dual_lambda = 10 * np.abs(np.random.rand(n_basis, 1))
    m, _, k = np.shape(X_hat)
    r = n_basis

    SS_t = np.zeros([r, r, k])
    XS_t = np.zeros([m, r, k])

    for kk in range(k):
        x_hat_k = X_hat[:, :, kk]
        s_hat_k = S_hat[:, :, kk]

        SS_t[:, :, kk] = np.matmul(s_hat_k, np.transpose(s_hat_k))
        XS_t[:, :, kk] = np.matmul(x_hat_k, np.transpose(s_hat_k))

    bnds = tuple([0, None] for _ in range(len(dual_lambda)))
    fun = lambda x: fobj_basis_dual(x, XS_t, SS_t, k)

    res = minimize(fun, dual_lambda, method='L-BFGS-B', bounds=bnds)

    LAMBDA = np.diag(res.x)
    B_hat = np.zeros([m, r, k])

    for kk in range(k):
        SS_t_k = np.squeeze(SS_t[:, :, kk])
        XS_t_k = np.squeeze(XS_t[:, :, kk])
        B_hat_k_t = np.matmul(np.linalg.pinv(SS_t_k + LAMBDA), np.transpose(XS_t_k))
        B_hat[:, :, kk] = np.transpose(B_hat_k_t)

    B = np.fft.ifft(B_hat, axis=-1)
    B[np.where(B == None)] = 0
    B = np.real(B)

    return B

def fobj_basis_dual(x, XS_t, SS_t, k):
    m = np.shape(XS_t)[1]
    r = len(x)
    LAMBDA = np.diag(x)

    f = 0
    # g = np.zeros([r, 1])
    # H = np.zeros([r, r])

    for kk in range(k):
        XS_t_k = XS_t[:, :, kk]
        SS_t_k = SS_t[:, :, kk]
        SS_t_inv = np.linalg.pinv(SS_t_k + LAMBDA)

        if m > r:
            f += np.trace(np.matmul(SS_t_inv, np.matmul(np.transpose(XS_t_k), XS_t_k)))
        else:
            f += np.trace(np.matmul(np.matmul(XS_t_k, SS_t_inv), np.transpose(XS_t_k)))

        f = np.real(f + k * np.sum(x))

    return f


if __name__ == '__main__':
    X_hat = np.random.rand(4, 3, 2)
    S = np.random.rand(2, 3, 2)
    n_basis = 2
    B = tensor_dl(X_hat, S, n_basis)
    print(B)
