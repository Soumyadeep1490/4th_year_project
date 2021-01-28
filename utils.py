#!/usr/bin/env python3

import numpy as np
from scipy.fftpack import dct, idct

# Discrete Cosine Transform
# here all the calculations and modifications are done according to the
# wikipedia entry of DCT
# https://en.wikipedia.org/wiki/Discrete_cosine_transform
#
# orthogonality feature helps us to apply DCT and inverse (IDCT) it very easily
# to make it orthoonal we have to modify the resultant array (see below)

# function to get DCT on a 1D array
def dct1D(x):
    '''
    this function takes an 1D array/vector as an input of dimention N and we
    apply the 1D DCT-II to get the DCT of the array/vector

    https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
        N-1
    Xk = ∑ xn cos[(n/𝝅) (n + ½) k]      k = 0,...., N - 1
        n=0

    to make it easily invertable we have to make it orthogonal and to do that
    we need to multily the X0 term with (1 / √2) and then multiply the
    resultant array elements with √(2 / N)

    this function returns the dct of the given 1D array/vector
    '''
    # check if the input array is 1D
    assert (len(x.shape) == 1), "ERROR[dct1D()]: NOT A 1D ARRAY!!"

    # get the dimention
    N = x.shape[0]

    # initializing resultant array/vector
    X = np.zeros((N))

    # populate the resuktant array/vector
    for k in range(N):
        a = 0
        for n in range(N):
            a += np.cos((np.pi / N) * (n + (1 / 2)) * k) * x[n]

        # handle the 0th term (special case)
        if k == 0:
            a *= np.sqrt(1 / 2)

        # modify all the elements to make it orthogonal
        X[k] = a * np.sqrt(2 / N)

    # return the resultant DCT transformed array/vector
    return X


# function to perform DCT on a matrix
def dct2D(x):
    '''
    this function takes a 2D array/matrix x with dimention N1 x N2 and performs
    the DCT transformation on that array/matrix

    https://en.wikipedia.org/wiki/Discrete_cosine_transform#M-D_DCT-II

    two-dimensional DCT-II of an image or a matrix is simply the
    one-dimensional DCT-II, from above, performed along the rows and then along
    the columns (or vice versa)

    as this is a 2D implementation it'll work only in one color channel
    so to get a dct transformation on a RGB image we have to call this function
    3 times with each of the 3 channels

    this function returns the DCT transformation of the given matrix
    '''
    # get the dimentions
    N1, N2 = x.shape

    # initialize the resultant DCT transformation of the given matrix
    X = np.zeros((N1, N2))

    # first perform 1D DCT along the rows of the matrix
    for k1 in range(N1):
        X[k1, :] = dct1D(x[k1, :])

    # then perform 1D DCT along the columns of the matrix generated from the
    # pervious step
    for k2 in range(N2):
        X[:, k2] = dct1D(X[:, k2])

    # round the results to 3 digits after decimal point
    return np.round(X, 3)


# function to perform IDCT on a 1D array
def idct1D(x):
    '''
    this function takes an 1D array/vector and performs the IDCT (DCT-III)
    transformation on the array/vector

    https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-III
              N-1
    Xk = ½x0 + ∑ xn cos[(𝝅/N) (k + ½) n]      k = 0,...., N - 1
              n=1

    to make it orthogonal we have devide the first term by √2 instead of 2 and
    then we have to multiply the resultant array/vector by √(2 / N)

    this function returns the IDCT transformation of the given array/vector
    '''
    # check if the input array is 1D
    assert (len(x.shape) == 1), "ERROR[idct1D()]: NOT A 1D ARRAY!!"

    # get the dimentions
    N = x.shape

    # initialize the resultant IDCT array
    X = np.zeros((N))

    # calculate the first term
    x0 = x[0] * np.sqrt(1 / 2)

    # populate the resultant IDCT array
    for k in range(N):
        a = x0
        for n in range(1, N):
            a += np.cos((np.pi / N) * n * (k + (1 / 2))) * x[n]

        # modify all the terms for orthogonality
        X[k] = a * np.sqrt(2 / N)

    # return the IDCT of the given array/vector
    return X


# function to perform IDCT(DCT-III) on a 2D array
def idct2D(x):
    pass

def dct2_ (block):
    '''
    implements the inbuilt dct function from the scipy module
    '''
    return dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')


np.random.seed(2021)

arr = np.random.randint(low=0, high=255, size=(3, 3, 3), dtype=np.uint8)
arr0 = arr[:, :, 0]
arr1 = arr[:, :, 1]
arr2 = arr[:, :, 2]

_F = np.zeros((3, 3, 3))
_F[:, :, 0] = dct2D(arr0)
_F[:, :, 1] = dct2D(arr1)
_F[:, :, 2] = dct2D(arr2)

F = np.zeros((3, 3, 3))
F[:, :, 0] = np.round(dct2_(arr0), 3)
F[:, :, 1] = np.round(dct2_(arr1), 3)
F[:, :, 2] = np.round(dct2_(arr2), 3)

if np.allclose(_F, F):
    print('SAME')
