#!/usr/bin/env python3

import numpy as np
from scipy.fftpack import dct, idct

# Discrete Cosine Transform
# for a 2D image DCT-II is given as f(x, y) => F(u, v)
# wiki: https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
# IDCT in matrix form: f = (C.T)FC | C.T -> C transpose
# if we can make this orthogonal then we can easily inverse this hence get IDCT
# very easily

# function to get the DCT coefficient matrix
def dct_mat(N):
    '''
    this function takes an image array as an input of dimention m x n and we
    apply the 2D DCT-II to get the DCT matrix

    C -> Cosine Transformation Matrix (orthogonal)
    C(u, v) = sqrt(1 / N) , where u = 0 and 1 <= v <= N - 1
            = sqrt(2 / N) * Cos(((2 * v + 1) * 𝜋 * u) / (2 * N))
                                   , where 1 <= u <= N - 1 and 1 <= v <= N - 1

    as this is a 2D implementation it'll work only in one color channel
    so to get a dct matrix on a RGB image we have to call this function 3 times
    with each of the 3 channels

    this function returns the dct matrix of the image array
    '''
    # DCT matrix initialization
    C = np.zeros((N, N))

    # populate the DCT matrix
    for m in range(N):
        for n in range(N):
            if m == 0:
                C[m, n] = np.sqrt(1 / N)
            else:
                C[m, n] =\
                np.sqrt(2 / N) * np.cos(((2 * n + 1) * np.pi * m) / (2 * N))


    return C


# function to perform DCT on a matrix
def _dct(mat):
    '''
    this function takes an image array and performs the DCT transformation on
    the matrix

    C -> DCT matrix of size N x N
    C.T -> C transpose
    DCT in matrix form: F = Cf(C.T)

    this function returns the DCT transformation of the given matrix
    '''
    # get the dimentions
    M, N = mat.shape

    # for now M = N
    assert (M == N), "NOT A SQUARE MATRIX!!"

    # get the DCT matrix of the dimention
    C = dct_mat(M)
    # get the transpose of the DCT matrix
    CT = C.T
    # get the DCT transformation
    out = np.matmul(np.matmul(C, mat), CT)

    # return the transformed matrix rounded to 3 digits after decimal
    return np.round(out, 3)


# function to perform IDCT on a matrix
def _idct(MAT):
    '''
    this function takes an image array and performs the IDCT transformation on
    the matrix

    C -> DCT matrix of size N x N
    C.T -> C transpose
    IDCT in matrix form: f = (C.T)FC

    this function returns the IDCT transformation of the given matrix
    '''
    # get the dimentions
    M, N = MAT.shape

    # for now M = N
    assert (M == N), "NOT A SQUARE MATRIX!!"

    # get the DCT matrix of the dimention
    C = dct_mat(M)
    # get the transpose of the DCT matrix
    CT = C.T
    # get the DCT transformation
    out = np.matmul(np.matmul(CT, MAT), C)

    # return the transformed matrix rounded to 3 digits after decimal
    return np.round(out, 3)


np.random.seed(2021)

arr = np.random.randint(low=0, high=255, size=(3, 3, 3), dtype=np.uint8)
arr0 = arr[:, :, 0]
arr1 = arr[:, :, 1]
arr2 = arr[:, :, 2]

# arr = np.eye(3)
arr = np.array([[1, 2], [3, 4]])
# _F = np.zeros((3, 3, 3))
# _F[:, :, 0] = dct_mat(arr0)
# _F[:, :, 1] = dct_mat(arr1)
# _F[:, :, 2] = dct_mat(arr2)
_F = _dct(arr)
print(_F)
print('------------------------')

_F = _idct(_F)
print(_F)
print('------------------------')
# F = np.zeros((3, 3, 3))
# F[:, :, 0] = np.round(dct(arr0, type=2, norm='ortho'), 3)
# F[:, :, 1] = np.round(dct(arr1, type=2, norm='ortho'), 3)
# F[:, :, 2] = np.round(dct(arr2, type=2, norm='ortho'), 3)
F = np.round(dct(arr, type=2, norm='ortho'), 3)
print(F)

F = np.round(idct(F, type=2, norm='ortho'), 3)
print(F)
# print(np.round(dct(arr, type=2, norm='ortho'), 3))

if np.allclose(_F, F):
    print('SAME')
