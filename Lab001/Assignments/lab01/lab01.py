# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import sys
from numpy.lib.stride_tricks import sliding_window_view

# Allow helpers package to be found when this module is imported standalone
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import load_image, get_data_path

SPHINX_IMAGE = "2560px-Great_Sphinx_of_Giza_-_20080716a.jpg"

def read_image():
    """Load and preprocess the Sphinx image: scale by 0.5 and convert to greyscale."""
    image_path = get_data_path(SPHINX_IMAGE)
    return load_image(image_path, scale_factor=2, as_gray=True)

# Skeleton Code will generate a 1D Gaussian Kernel for given sigma.
# You need to implement 2D convolution as efficiently as possible.
class GaussianFilt:
    def __init__(self, sigma):
        self.sigma = sigma

    def gauss_kernel(self):
        """Generate a normalised 1D Gaussian kernel of appropriate size for self.sigma.

        Returns:
            gauss_1d (ndarray): Shape (k_size, 1) — column vector kernel.
        """
        sigma = self.sigma
        k_size = int(6 * sigma + 1)
        half = k_size // 2
 
        x = np.arange(k_size) - half
        g = np.exp(-(x ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
 
        gauss_1d = np.expand_dims(g, axis=1).astype(np.float64)
        return gauss_1d
 

    def my_conv_method(self, image):
        """Perform separable 2D Gaussian convolution via two 1D matrix-multiply passes.

        Args:
            image (ndarray): 2D greyscale image (H, W), any numeric dtype.

        Returns:
            conv_img (ndarray): Blurred image of same shape, dtype uint8.
        """
        kernel = self.gauss_kernel()
        k = kernel.shape[0]
        pad = k // 2
        k_vec = kernel.ravel()
 
        img = image.astype(np.float64)
        padded = np.pad(img, pad, mode="symmetric")
 
        windows_h = sliding_window_view(padded, k, axis=1)
        horiz = windows_h @ k_vec
 
        windows_v = sliding_window_view(horiz, k, axis=0)
        vert = windows_v @ k_vec
 
        conv_img = np.clip(vert, 0, 255).astype(np.uint8)
        return conv_img
