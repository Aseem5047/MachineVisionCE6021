# -*- coding: utf-8 -*-
"""Lab 02 — MRI Noise Removal via Fourier Notch Filtering.

The corrupted MRI scan is supplied as centred (FFT-shifted) k-space data.
A periodic noise pattern shows up as a pair of bright spikes in that data.
You will remove the noise in five small steps:

  1. fourier_magnitude(f_shifted)       view the spectrum
  2. reconstruct(f_shifted)             turn k-space back into an image
  3. get_noise_peaks()                  the spike positions, read off the plot
  4. make_notch_mask(centres, radius)   a mask that zeroes a disc at each spike
  5. remove_noise(f_shifted)            put the steps together

Run evaluate_lab02.py after each step to see your progress.
All positions in this lab are written as (row, col).
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from helpers.mridataprocessor import MRIDataProcessor  # noqa: F401

class MRIprocessor:
    """MRI noise removal via Fourier-domain notch filtering."""

    def __init__(self, shape=(256, 256)):
        self.shape = shape

    def fourier_magnitude(self, f_shifted):
        """Step 1 — log-magnitude of the k-space data, for viewing.

        Args:
            f_shifted (np.ndarray): 2D complex, centred k-space data.

        Returns:
            np.ndarray: 2D float array, log(|f_shifted| + 1).
        """
        raise NotImplementedError("Implement this method")

    def reconstruct(self, f_shifted):
        """Step 2 — convert centred k-space data back into a spatial image.

        Args:
            f_shifted (np.ndarray): 2D complex, centred k-space data.

        Returns:
            np.ndarray: 2D float image (magnitude), centred.
        """
        raise NotImplementedError("Implement this method")

    def get_noise_peaks(self):
        """Step 3 — the (row, col) positions of the two noise spikes.

        Find these by hovering over the spectrum plot made by
        evaluate_lab02.py, then type the positions in here.

        Returns:
            list of (int, int): [(row, col), (row, col)]
        """
        raise NotImplementedError("Implement this method")

    def make_notch_mask(self, centres, radius=2):
        """Step 4 — a mask of ones with a zeroed disc around each centre.

        Args:
            centres (list of (int, int)): (row, col) positions to notch out.
            radius (int): Radius of each disc, in pixels (default 2).

        Returns:
            np.ndarray: float array of shape self.shape, values 0 or 1.
        """
        raise NotImplementedError("Implement this method")

    def remove_noise(self, f_shifted):
        """Step 5 — remove the noise using the methods above.

        Args:
            f_shifted (np.ndarray): 2D complex, centred k-space data.

        Returns:
            tuple: (filtered_img, corrupted_img, fourier_img, mask)
        """
        raise NotImplementedError("Implement this method")
