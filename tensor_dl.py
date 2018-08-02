# -*- coding:utf-8 -*-
# 
# Author: miaoyin
# Time: 2018/8/2 22:11

import numpy as np


def tensor_dl(X_hat, S, num_bases):
    S_hat = np.fft.fftn(S, axes=3) #mabe incorrect
    dual_lambda = 10 * np.abs(np.random.rand(num_bases, 1))
    m, _, k = np.shape(X_hat)
    r = num_bases

    SS_t = np.zeros([r, r, k])
    XS_t = np.zeros([m, r, k])

    for kk in range(k)
        x_hat_k = X_hat[:,:,kk]
        s_hat_k = S_hat[:,:,kk]

        SS_t[:,:,kk] = s_hat_k * np.transpose(s_hat_k)
        XS_t[:,:,kk] = x_hat_k * np.transpose(x_hat_k)


    lb = np.zeros(np.shape(dual_lambda))
    options = op
