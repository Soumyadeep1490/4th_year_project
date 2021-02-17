#!/usr/bin/env python3
import numpy as np
from scipy.fftpack import dct, idct

# Discrete Cosine Transform
# this is a faster implementation of a faster DCT algorithm than the standard
# one
# http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.463.3353&rep=rep1&type=pdf
# https://web.stanford.edu/class/ee398a/handouts/lectures/07-TransformCoding.pdf#page=30
# it olny requires 5+8 multiplications and 29 additions on a vector of length 8

# function to implement dct faster
def dct1D(x):
    '''
    this function takes an vector x of length 8 as input and the dct algorithm
    is applied to it

    # Arai, Agui and Nakajama (AAN) DCT Algorithm
    # https://web.stanford.edu/class/ee398a/handouts/lectures/07-TransformCoding.pdf#page=30

    this function returns the dct of the given array
    '''
    # get the length of the array
    n = len(x)

    # check the input length is 8 at maximum
    assert (n <= 8), "ERROR! Must have less than 8 elements"

    # if the length is less than 8 zero pad the rest
    x = np.pad(x, (0, 8 - n), 'constant')

    # required constant vectors
    # Ck = cos((𝝅 / 16) * k)    # k = 0,..., 7
    c = [np.cos((np.pi / 16) * k) for k in range(8)]
    # Sk = 1 / (4 * Ck)         # k = 1,..., 7
    s = [(1 / (4 * c[k])) for k in range(8)]
    # S0 = 1 / (2√2)
    s[0] = 1 / (2 * (2 ** 0.5))

    a = [c[4], c[2] - c[6], c[4], c[6] + c[2], c[6]]

    # NOTE: names are given as: f[stage index][signal index] | 0-based indexing
    # eg:
    #     o/p of x0 at 2nd stage -> f10
    # [+] o/p of x5 at 3rd stage -> f25
    # ---------------------------------------------
    # produces x5 at 4th stage   -> f35 = f10 + f25

    # first stage computations
    f00 = x[0] + x[7]            # x0
    f01 = x[1] + x[6]            # x1
    f02 = x[2] + x[5]            # x2
    f03 = x[3] + x[4]            # x3
    f04 = x[3] - x[4]            # x4
    f05 = x[2] - x[5]            # x5
    f06 = x[1] - x[6]            # x6
    f07 = x[0] - x[7]            # x7

    # second stage computations
    f10 = f00 + f03              # x0
    f11 = f01 + f02              # x1
    f12 = f01 - f02              # x2
    f13 = f00 - f03              # x3
    f14 = -(f05 + f04)           # x4
    f15 = f05 + f06              # x5
    f16 = f06 + f07              # x6

    # thrid stage computatiions
    f20 = f10 + f11              # x0
    f21 = f10 - f11              # x1
    f22 = f12 + f13              # x2

    # fourth stage computations
    f32 = f22 * a[0]            # x2
    f34 = f14 * a[1]            # x4
    f35 = f15 * a[2]            # x5
    f36 = f16 * a[3]            # x6
    tmp = (f14 + f16) * a[4]

    # fifth stage computations
    f44 = -(f34 + tmp)          # x4
    f46 = f36 - tmp             # x6

    # sixth stage computations
    f52 = f32 + f13             # x2
    f53 = f13 - f32             # x3
    f55 = f35 + f07             # x5
    f57 = f07 - f35             # x7

    # seventh stage computations
    f64 = f44 + f57             # x4
    f65 = f55 + f46             # x5
    f66 = f55 - f46             # x6
    f67 = f57 - f44             # x7

    # eighth stage computations
    f70 = round(f20 * s[0], 8)  # x0
    f71 = round(f21 * s[4], 8)  # x1
    f72 = round(f52 * s[2], 8)  # x2
    f73 = round(f53 * s[6], 8)  # x3
    f74 = round(f64 * s[5], 8)  # x4
    f75 = round(f65 * s[1], 8)  # x5
    f76 = round(f66 * s[7], 8)  # x6
    f77 = round(f67 * s[3], 8)  # x7

    # rearrange the final components in the correct order to get the dct
    return [f70, f75, f72, f77, f71, f74, f73, f76]



def idct1D(x):
    '''
    this function takes a vector of length 8 and applies inverse dct(idct) to
    obtain the original vector

    for AAN algorithm(see dct1D section) we just have to perform the operations
    in reverse to get the inverse of the dct

    this function returns the idct of the given vector
    '''
    # check if the input length is 8 at maimum
    assert(len(x) <= 8), "ERROR! Must have less than 8 elements"

    # if the length is less than 8 zero pad the rest
    x = x + [0] * (8 - len(x))
    # x += x[:8 - len(x)]

    # required constant vectors
    # Ck = cos((𝝅 / 16) * k)    # k = 0,..., 7
    c = [np.cos((np.pi / 16) * k) for k in range(8)]
    # Sk = 1 / (4 * Ck)         # k = 1,..., 7
    s = [(1 / (4 * c[k])) for k in range(8)]
    # S0 = 1 / (2√2)
    s[0] = 1 / (2 * (2 ** 0.5))

    a = [c[4], c[2] - c[6], c[4], c[6] + c[2], c[6]]

    # NOTE: names are given as: f[stage index][signal index] | 0-based indexing
    # eg:
    #     o/p of x0 at 2nd stage -> f10
    # [+] o/p of x5 at 3rd stage -> f25
    # ---------------------------------------------
    # produces x5 at 4th stage   -> f35 = f10 + f25

    f20 = x[0] / s[0]            # x0
    f21 = x[4] / s[4]            # x1
    f52 = x[2] / s[2]            # x2
    f53 = x[6] / s[6]            # x3
    f64 = x[5] / s[5]            # x4
    f65 = x[1] / s[1]            # x5
    f66 = x[7] / s[7]            # x6
    f67 = x[3] / s[3]            # x7

    f44 = (f64 - f67) / 2       # x4
    f55 = (f65 + f66) / 2       # x5
    f46 = (f65 - f66) / 2       # x6
    f57 = (f64 + f67) / 2       # x7

    f32 = (f52 - f53) / 2       # x2
    f13 = (f52 + f53) / 2       # x3
    f35 = (f55 - f57) / 2       # x5
    f07 = (f55 + f57) / 2       # x7

    f10 = (f20 + f21) / 2       # x0
    f11 = (f20 - f21) / 2       # x1

    tmp = (f44 - f46) * a[4]
    f22 = f32 / a[0]            # x2
    f14 = (f44 * a[3] - tmp) / (a[1] * a[4] - a[1] * a[3] - a[3] * a[4]) # x4
    f15 = f35 / a[2]            # x5
    f16 = (tmp - f46 * a[1]) / (a[1] * a[4] - a[1] * a[3] - a[3] * a[4]) # x6

    f06 = f16 - f07             # x6
    f05 = f15 - f06             # x5
    f04 = -(f05 + f14)          # x4
    f12 = f22 - f13             # x2

    f00 = (f10 + f13) / 2       # x0
    f01 = (f11 + f12) / 2       # x1
    f02 = (f11 - f12) / 2       # x2
    f03 = (f10 - f13) / 2       # x3

    x0 = round((f00 + f07) / 2, 1) # x0
    x1 = round((f01 + f06) / 2, 1) # x1
    x2 = round((f02 + f05) / 2, 1) # x2
    x3 = round((f03 + f04) / 2, 1) # x3
    x4 = round((f03 - f04) / 2, 1) # x4
    x5 = round((f02 - f05) / 2, 1) # x5
    x6 = round((f01 - f06) / 2, 1) # x6
    x7 = round((f00 - f07) / 2, 1) # x7

    # return the reconstructed array
    return [int(x0), int(x1), int(x2), int(x3), int(x4), int(x5), int(x6),
            int(x7)]



# TODO
def dct2D(x):
    pass

def idct2D(x):
    pass

def dct3D(x):
    pass

def idct3D(x):
    pass
