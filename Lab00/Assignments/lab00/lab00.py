# -*- coding: utf-8 -*-
"""Lab 00 — Introduction: Simple Image Processing.

Welcome to your first lab!  The goal is to get comfortable with the lab
workflow: implement a class in this file, run an evaluation script to see your
results.

Task
----
Implement the two methods inside the SimpleImageProcessing class:

  add_blur(image, **kwargs)
      Apply Gaussian blur to image.  Use the 'ksize' keyword argument
      (default 15) to control the kernel size (must be a positive odd integer).

  add_sharpen(image, **kwargs)
      Sharpen image using an unsharp-mask approach (subtract a blurred version
      from the original, scaled by a 'strength' keyword argument (default 1.5)).

Both methods should return the processed image as a uint8 numpy array with the
same shape as the input.
"""
import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.dataloader import get_data_path, load_image  # noqa: F401

class SimpleImageProcessing:
    """A simple image-processing engine with blur and sharpening methods.

    No constructor arguments are needed — just instantiate and call the methods.

    Example
    -------
    >>> proc = SimpleImageProcessing()
    >>> blurred   = proc.add_blur(image, ksize=21)
    >>> sharpened = proc.add_sharpen(image, strength=2.0)
    """

    def add_blur(self, image, **kwargs):
        """Apply Gaussian blur to an image.

        Args:
            image  (ndarray): H × W × C or H × W input image (uint8).
            **kwargs:
                ksize (int): Gaussian kernel size — must be a positive odd
                             integer (default 15).

        Returns:
            ndarray: Blurred image, same shape and dtype as input.
        """
        raise NotImplementedError("Implement this method")

    def add_sharpen(self, image, **kwargs):
        """Sharpen an image using an unsharp mask.

        Subtracts a blurred version from the original, scaled by 'strength',
        to enhance high-frequency detail.

        Args:
            image  (ndarray): H × W × C or H × W input image (uint8).
            **kwargs:
                ksize    (int):   Gaussian kernel size for the mask (default 15).
                strength (float): How strongly to apply the sharpening (default 1.5).

        Returns:
            ndarray: Sharpened image, same shape and dtype as input.
        """
        raise NotImplementedError("Implement this method")
