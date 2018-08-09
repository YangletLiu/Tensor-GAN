import numpy as np
from params import Params as params


def tensor_tsta(X, D0, B0):

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
        fobj[i] = obj_fun(X, D0, B1)
    B = B1

    return B, fobj

def obj_fun(X, D, B):
    diff = X - tensor_product(D, '', B, '')
    fobj = 0.5 * np.linalg.norm(diff) ** 2 + params.beta * np.sum(np.abs(B))

    return fobj

def blk_circ_mat(A):
    sz_A = np.shape(A)
    dim = np.zeros(1,2)
    dim[0] = sz_A[0] * sz_A[2]
    dim[1] = sz_A[1] * sz_A[2]
    A_C = np.zeros(dim)
    A_mat = np.reshape(np.transpose(A, [0, 2, 1]), [dim[0], sz_A[1]])
    A_c[:,:sz_A[1]] = A_mat
    for k in range(1, sz_A[2]):
        A_c[:,(k-1)*sz_A[1]+1:k*sz_A[1]] = np.roll(A_mat, (k-1)*sz_A[0], axis=0)

    return A_c

if __name__ == '__main__':
    X = np.random.rand(4,3,2)
    D0 = np.random.rand()
    a = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    print(np.mat(a))
    print(np.roll(a,1,axis=0))

