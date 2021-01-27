#!/usr/bin/env python3

import numpy as np
from scipy.fftpack import dct

# Discrete Cosine Transform
# for a 2D image DCT-II is given as f(x, y) => F(u, v)
# wiki: https://en.wikipedia.org/wiki/Discrete_cosine_transform#DCT-II
# In matrix form: F = Cf(C.T) | C.T -> C transpose

def _dct(arr):
    '''
    this function takes an array as an input of dimention m x n and we apply
    the DCT-II transformation to it

    C -> Cosine Transformation Matrix
    C(u, v) = sqrt(1 / N) , where u = 0 and 1 <= v <= N - 1
            = sqrt(2 / N) * Cos(((2 * v + 1) * 𝜋 * u) / (2 * N))
                                   , where 1 <= u <= N - 1 and 1 <= v <= N - 1

    this function returns the dct transform of the array
    '''
    # get the dimentions of the array (currently 2D)
    m, n = arr.shape

    # DCT matrix initialization
    C = np.zeros((m, n))

    # populate the DCT matrix
    for u in range(m):
        for v in range(n):
            if u == 0:
                C[u, v] = np.sqrt(1 / m)
            else:
                C[u, v] =\
                np.sqrt(2 / m) * np.cos(((2 * v + 1) * np.pi * u) / (2 * m))


    return np.round(C.T, 3)

arr = np.eye(7)
_F = _dct(arr)
print(_F)

F = np.round(dct(arr, type=2, norm='ortho'), 3)
print(F)

if np.allclose(_F, F):
    print('SAME')
