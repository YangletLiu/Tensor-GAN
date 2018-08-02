import numpy as np


def tensor_tsta(X, D0, B0)

    _, n, k = np.shape(X)

    D0_t_D0 = tensor_product(D0, 't', D0, [])

    D0_c = blk_circ_mat(D0_t_D0)
    L0 = norm(D0_c)

    D0_t_X = tensor_product(D0, 't', X, [])

    C1 = B0
    t1 = 1
    fobj = np.zeros([params.r, n, k])

    for i in range(max_iter):
        L1 = params.eta ** i * L0
        grad_C1 = tensor_product(D0_t_D0, 't', C1, []) - D0_t_X
        temp = C1 - grad_C1 / L1
        B1 = np.sign(temp) * max(abs(temp) - params.beta / L1, 0)
        t2 = (1 + np.sqrt(1 + 4 * t1 ** 2)) / 2
        C1 = B1 + ((t1 - 1) / t2) * (B1 - B0)
        B0 = B1
        t1 = t2
        fobj(i) = obj_fun(X, D0, B1)
    B = B1

    return B, fobj

def obj_fun(X, D, B)
    diff = X - tensor_product(D, '', B, '')
    fobj = 0.5 * np.linalg.norm(diff) ** 2 + params.beta * np.sum(np.abs(B))

    return fobj

