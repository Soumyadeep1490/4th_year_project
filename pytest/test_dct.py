#!/usr/bin/env python3

# This is the testing script for DCT and IDCT functions only

import unittest
import numpy as np
from utils import *
from scipy.fftpack import dct, idct # to evaluate our results

# set the seed for consistency
np.random.seed(2021)

def dct2 (block):
    '''
    implements the inbuilt dct function from the scipy module
    '''
    return dct(dct(block.T, type=2, norm = 'ortho').T, type=2, norm = 'ortho')


class TestDCT(unittest.TestCase):
    '''
    unittest class to check if the DCT and IDCT functions are working properly
    '''
    def test_dct1D(self):
        # 1D array of length 2
        arr = np.array([64, 39])
        # manual dct1D()
        res = dct1D(arr)
        # builtin dct1D() by scipy
        goal = dct(arr, type=2, norm='ortho')
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct1D() of length 2 FAIL!"

        # 1D array of length 50
        arr = np.random.randint(low=9, high=9999, size=50)
        # manual dct1D()
        res = dct1D(arr)
        # builtin dct1D() by scipy
        goal = dct(arr, type=2, norm='ortho')
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct1D() of length 50 FAIL!"

        # 1D array of length 100
        arr = np.random.randint(low=-9999, high=9999, size=100)
        # manual dct1D()
        res = dct1D(arr)
        # builtin dct1D() by scipy
        goal = dct(arr, type=2, norm='ortho')
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct1D() of length 100 FAIL!"

    def test_dct2D(self):
        # 2D array of dimention 2 x 2
        arr = np.array([[64, 39], [27, 82]])
        # manual dct2D()
        res = dct2D(arr)
        # builtin dct2D() by scipy
        goal = dct2(arr)
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct2D() of shape 2x2 FAIL!"

        # 2D array of dimention 50 x 50
        arr = np.random.randint(low=9, high=9999, size=(50, 50))
        # manual dct2D()
        res = dct2D(arr)
        # builtin dct2D() by scipy
        goal = dct2(arr)
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct2D() of shape 50x50 FAIL!"

        # 2D array of dimention 30 x 70
        arr = np.random.randint(low=-9999, high=9999, size=(30, 70))
        # manual dct2D()
        res = dct2D(arr)
        # builtin dct2D() by scipy
        goal = dct2(arr)
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct2D() of shape 30x70 FAIL!"

        # check for identity matrix
        arr = np.eye(15)
        # manual dct2D()
        res = dct2D(arr)
        # builtin dct2D() by scipy
        goal = dct2(arr)
        # check if they are same
        assert (np.allclose(res, goal)), "test_dct2D() of identity matrix FAIL!"


