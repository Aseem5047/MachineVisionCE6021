# -*- coding: utf-8 -*-
"""Lab 00B — Introduction: Gaussian Kernels, Convolution & Interpolation.

Building on Lab 00's class/method workflow, this lab gets you comfortable
with three numpy/skimage building blocks used throughout the rest of the
series: constructing a filter kernel from a 1D formula, applying it as a
2D convolution, and resampling an image with interpolation.

Task
----
Implement the three methods inside the ConvolutionInterpolator class:

  gaussian_kernel_2d(sigma)
      Build a 2D Gaussian kernel from the outer product of a 1D Gaussian.
      The kernel size K is not a separate input — it's derived from sigma
      using the ±3σ rule of thumb.

  convolve(image, kernel, **kwargs)
      Convolve a 2D image with a kernel using scipy.signal.convolve2d,
      using the 'mode' keyword argument (default 'same') to control the
      output size and 'boundary' (default 'symm') to control padding.

  interpolate(image, **kwargs)
      Resample a 2D image using skimage.transform.rescale, using the
      'scale_factor' (default 0.5) and 'order' (default 0) keyword
      arguments. Anti-aliasing must stay switched off — this method should
      perform interpolation ONLY, with no implicit smoothing of its own.
"""
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale

class ConvolutionInterpolator:
    """Small helper class for building Gaussian kernels and applying
    convolution / interpolation to 2D image arrays.

    No constructor arguments are needed — just instantiate and call the
    methods below.

    Example Usage
    -------------
    >>> ci = ConvolutionInterpolator()
    >>> kernel = ci.gaussian_kernel_2d(sigma=2.5)
    >>> smoothed = ci.convolve(image, kernel)
    >>> small = ci.interpolate(smoothed, scale_factor=0.25)
    """

    def gaussian_kernel_2d(self, sigma):
        """Build a 2D Gaussian kernel from the outer product of a 1D Gaussian.

        The kernel size K is derived from sigma via the rule of thumb that
        a Gaussian's mass is concentrated within about ±3σ of its centre
        (see README) — it is not a separate input.

        Args:
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            ndarray: K x K kernel, dtype float, normalised to sum to 1.
        """
        raise NotImplementedError("Implement this method")

    def convolve(self, image, kernel, **kwargs):
        """Convolve a 2D image with a kernel.

        Args:
            image  (ndarray): H x W greyscale image.
            kernel (ndarray): Q x Q filter kernel, e.g. from gaussian_kernel_2d.
            **kwargs:
                mode      (str): Output size passed to scipy.signal.convolve2d
                                  — 'same', 'valid', or 'full' (default 'same').
                boundary  (str): Boundary/padding style — 'fill', 'wrap', or
                                  'symm' (default 'symm').
                fillvalue (float): Constant used when boundary='fill'
                                    (default 0).

        Returns:
            ndarray: Filtered image — same shape as the input under the
            default mode='same', or a different shape for 'valid'/'full'.
        """
        raise NotImplementedError("Implement this method")

    def interpolate(self, image, **kwargs):
        """Resample a 2D image by a given scale factor.

        Args:
            image (ndarray): H x W greyscale image.
            **kwargs:
                scale_factor (float): Output size relative to input, e.g.
                                       0.25 downsamples to a quarter size
                                       (default 0.5).
                order        (int):   Interpolation order — 0 = nearest
                                       neighbour, 1 = bilinear, 3 = bicubic,
                                       etc. (default 0).

        Returns:
            ndarray: Resampled image, shape scaled by scale_factor.
        """
        raise NotImplementedError("Implement this method")
