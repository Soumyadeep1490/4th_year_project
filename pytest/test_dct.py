#!/usr/bin/env python3

# This is the testing script for DCT and IDCT functions only

import unittest
import numpy as np
from utils import *

# set the seed for consistency
np.random.seed(2021)


class TestDCT(unittest.TestCase):
    '''
    unittest class to check if the DCT and IDCT functions are working properly
    '''
    def test_1D(self):
        # 1D array of length 2
        arr = np.array([64, 39])
        # dct1D()
        transform = dct1D(arr)
        # idct1D()
        inverse_transform = idct1D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform[:2])), "test_1D() of length 2 FAIL!"

        # 1D array of length 50
        arr = np.random.randint(low=9, high=9999, size=50)
        # dct1D()
        transform = dct1D(arr)
        # idct1D()
        inverse_transform = idct1D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform[:50])), "test_1D() of length 50 FAIL!"

        # 1D array of length 100
        arr = np.random.randint(low=-9999, high=9999, size=100)
        # dct1D()
        transform = dct1D(arr)
        # idct1D()
        inverse_transform = idct1D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform[:100])), "test_1D() of length 100 FAIL!"

        # 1D array of length 10000
        arr = np.random.randint(low=-9999, high=9999, size=10000)
        # dct1D()
        transform = dct1D(arr)
        # idct1D()
        inverse_transform = idct1D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform[:10000])), "test_1D() of length 10000 FAIL!"

    def test_2D(self):
        # 2D array of dimention 2x2
        arr = np.array([[64, 39],
                        [27, 82]])
        # dct2D()
        transform = dct2D(arr)
        # idct1D()
        inverse_transform = idct2D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_2D() of shape 2x2 FAIL!"

        # 2D array of dimention 50x50
        arr = np.random.randint(low=9, high=9999, size=(50, 50))
        # dct2D()
        transform = dct2D(arr)
        # idct1D()
        inverse_transform = idct2D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_2D() of shape 50x50 FAIL!"

        # 2D array of dimention 30x70
        arr = np.random.randint(low=-9999, high=9999, size=(30, 70))
        # dct2D()
        transform = dct2D(arr)
        # idct1D()
        inverse_transform = idct2D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_2D() of shape 30x70 FAIL!"

        # check for identity matrix (15x15)
        arr = np.eye(15)
        # dct2D()
        transform = dct2D(arr)
        # idct1D()
        inverse_transform = idct2D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_2D() of shape 15x15 FAIL!"

    def test_3D(self):
        # 3D array of dimention 2x2x2
        arr = np.array([[[64, 39],
                         [27, 82]],

                        [[45, 99],
                         [73, 2 ]]])
        # manual dct3D()
        transform = dct3D(arr)
        # idct1D()
        inverse_transform = idct3D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_3D() of shape 2x2x2 FAIL!"

        # 3D array of dimention 27x83x32
        arr = np.random.randint(low=-9999, high=9999, size=(27, 83, 32))
        # manual dct3D()
        transform = dct3D(arr)
        # idct1D()
        inverse_transform = idct3D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_3D() of shape 27x83x32 FAIL!"

        # test for simulated 1024x720x3 image
        arr = np.random.randint(low=0, high=256,
                                size=(1024, 720, 3), dtype=np.uint8)
        # manual dct3D()
        transform = dct3D(arr)
        # idct1D()
        inverse_transform = idct3D(transform)
        # check if they are same
        assert (np.allclose(arr, inverse_transform)), "test_3D() of shape 1024x720x3 FAIL!"


