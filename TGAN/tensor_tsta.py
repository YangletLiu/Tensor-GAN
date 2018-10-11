import numpy as np
from hyper_params import HyperParams as params
from tensor_product import tensor_product


def tensor_tsta(X, D, C0):

    _, n, k = np.shape(X)

    D_t_D = tensor_product(D, 't', D, '')

    D_c = blk_circ_mat(D_t_D)
    l0 = np.linalg.norm(D_c, 2)

    D_t_X = tensor_product(D, 't', X, '')

    J1 = C0
    t1 = 1
    #fobj = np.zeros(params.max_iter)

    for i in range(params.tsta_max_iter):
        l1 = params.eta ** i * l0
        grad_J1 = tensor_product(D_t_D, 't', J1, '') - D_t_X
        temp1 = J1 - grad_J1 / l1
        temp2 = np.abs(temp1) - params.beta / l1
        temp2[np.where(temp2 < 0)] = 0
        C1 = np.multiply(np.sign(temp1), temp2)
        t2 = (1 + np.sqrt(1 + 4 * t1 ** 2)) / 2
        J1 = C1 + ((t1 - 1) / t2) * (C1 - C0)
        C0 = C1
        t1 = t2
        #fobj[i] = obj_fun(X, D0, B1)
    C = C1

    return C

def obj_fun(X, D, C):
    diff = X - tensor_product(D, '', C, '')
    fobj = 0.5 * np.linalg.norm(diff) ** 2 + params.beta * np.sum(np.abs(C))

    return fobj

def blk_circ_mat(A):
    sz_A = np.shape(A)
    dim = [0, 0]
    dim[0] = sz_A[0] * sz_A[2]
    dim[1] = sz_A[1] * sz_A[2]
    A_c = np.zeros(dim)
    A_mat = np.reshape(np.transpose(A, [0, 2, 1]), [dim[0], sz_A[1]], order='F')
    for k in range(sz_A[2]):
        A_c[:,k*sz_A[1]:(k+1)*sz_A[1]] = np.roll(A_mat, k*sz_A[0], axis=0)

    return A_c

if __name__ == '__main__':
    s = np.random.rand(4,4)
    X = np.random.rand(4,3,2)
    D = np.random.rand(4,2,2)
    C0 = np.zeros([2,3,2])
    tensor_tsta(X,D,C0)
    import scipy.io as sio
    D0 = sio.loadmat('./samples/D0.mat')['D0']
    print(blk_circ_mat(D0).shape)
    D0_c = blk_circ_mat(D0)
    print(D0_c[100,:])
    print(np.linalg.norm(D0_c, 2))



