# -*- coding: utf-8 -*-
""" Lab 00 — Introduction: Simple Image Processing.

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
    """ A simple image-processing engine with blur and sharpening methods.

    No constructor arguments are needed — just instantiate and call the methods.

    Example
    -------
    >>> proc = SimpleImageProcessing()
    >>> blurred   = proc.add_blur(image, ksize=21)
    >>> sharpened = proc.add_sharpen(image, strength=2.0)
    """

    def add_blur(self, image, **kwargs):

        """
        Apply Gaussian blur to an image.

        Args:
            image  (ndarray): H × W × C or H × W input image (uint8).
            **kwargs:
                ksize (int): Gaussian kernel size — must be a positive odd
                             integer (default 15).

        Returns:
            ndarray: Blurred image, same shape and dtype as input.
        """

        # Pseudocode

        '''
        def add_blur(self, image, **kwargs):

            # 1. Get ksize from kwargs, default to 15 if not provided
            ksize = <look it up in kwargs with a default>

            # 2. Call cv2's Gaussian blur function
            #    - needs the image
            #    - needs kernel size as a tuple (ksize, ksize) — GaussianBlur wants
            #      width and height of the kernel separately, even though they're
            #      the same number here
            #    - needs a sigma value — pass 0 to let OpenCV auto-compute it
            blurred = <call cv2.GaussianBlur(...)>

            # 3. Return it — GaussianBlur already outputs uint8 when the input
            #    is uint8, so no clip/cast needed here
            return blurred
        '''
        
        # Read the kernel size from kwargs; fall back to the default of 15
        ksize = kwargs.get("ksize", 15)

        # GaussianBlur needs an odd kernel size (the kernel needs a centre
        # pixel), so round an even value up to the next odd number (20 -> 21)
        if ksize % 2 == 0:
            ksize += 1

        # Blur with a (ksize x ksize) kernel. sigma=0 lets OpenCV derive the
        # Gaussian's standard deviation from the kernel size
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # uint8 in -> uint8 out, so no clip/cast is needed here
        return blurred

    def add_sharpen(self, image, **kwargs):

        """
        Sharpen an image using an unsharp mask.

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

        # Pseudocode

        '''
        def add_sharpen(self, image, **kwargs):
            
            # 1. Get ksize (default 15) and strength (default 1.5) from kwargs
            ksize = <...>
            strength = <...>

            # 2. Get a blurred version of the image.
            #    Two ways to do this — either is fine:
            #      a) call cv2.GaussianBlur directly again, same as in add_blur
            #      b) call self.add_blur(image, ksize=ksize) to reuse your own method
            blurred = <...>

            # 3. Combine original + blurred using the unsharp mask weights:
            #    sharpened = (1 + strength) * image - strength * blurred
            #    cv2.addWeighted(src1, alpha, src2, beta, gamma) computes:
            #        alpha*src1 + beta*src2 + gamma
            #    so: src1=image, alpha=(1+strength), src2=blurred, beta=-strength, gamma=0
            sharpened = <call cv2.addWeighted(...)>

            # 4. Clip to valid pixel range BEFORE casting dtype
            sharpened = <np.clip(..., 0, 255)>

            # 5. Cast to uint8 (must come after clipping, not before)
            sharpened = <.astype(np.uint8)>

            return sharpened
        '''

        # Read the blur kernel size and the sharpening gain from kwargs,
        # falling back to their defaults (15 and 1.5)
        ksize = kwargs.get("ksize", 15)
        strength = kwargs.get("strength", 1.5)

        # Same odd-kernel rule as in add_blur: round an even ksize up by one
        if ksize % 2 == 0:
            ksize += 1

        # Blurred copy = the low-frequency content of the image.
        # (Equivalent alternative: self.add_blur(image, ksize=ksize))
        blurred = cv2.GaussianBlur(image, (ksize, ksize), 0)

        # Unsharp mask in one call:
        #   sharpened = (1 + strength) * image - strength * blurred
        #             = image + strength * (image - blurred)
        # i.e. add the removed high-frequency detail back, scaled by strength
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)

        # The weighted sum can leave [0, 255], so clip to the valid pixel range
        sharpened = np.clip(sharpened, 0, 255)

        # Cast to uint8 AFTER clipping (a uint8 cast wraps out-of-range values
        # instead of capping them)
        sharpened = sharpened.astype(np.uint8)

        return sharpened