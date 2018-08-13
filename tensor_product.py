import numpy as np

def tensor_product(A, ch1, B, ch2):
    dim = [0, 0, 0]
    sz_A = np.shape(A)
    sz_B = np.shape(B)
    dim[2] = sz_A[2]

    if ch1 == 't':
        dim[0] = sz_A[1]
    else:
        dim[0] = sz_A[0]

    if ch2 == 't':
        dim[1] = sz_B[0]
    else:
        dim[1] = sz_B[1]


    C_hat = np.zeros(dim)
    A_hat = np.fft.fft(A, axis=-1)
    B_hat = np.fft.fft(B, axis=-1)

    if ch1 == 't' and ch2 == 't':
        for k in range(dim[2]):
            C_hat[:,:,k] = np.matmul(np.transpose(A_hat[:,:,k]), np.transpose(B_hat[:,:,k]))
    elif ch1 == 't':
        for k in range(dim[2]):
            C_hat[:,:,k] = np.matmul(np.transpose(A_hat[:,:,k]), B_hat[:,:,k])
    elif ch2 == 't':
        for k in range(dim[2]):
            C_hat[:,:,k] = np.matmul(A_hat[:,:,k], np.transpose(B_hat[:,:,k]))
    else:
        for k in range(dim[2]):
            C_hat[:,:,k] = np.matmul(A_hat[:,:,k], B_hat[:,:,k])

    C = np.real(np.fft.ifft(C_hat, axis=-1))

    return C
