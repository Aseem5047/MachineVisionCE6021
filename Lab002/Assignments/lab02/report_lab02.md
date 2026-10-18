# Lab Report — Lab 02: MRI Noise Removal via Fourier Notch Filtering

## Overview

The goal of this lab was to remove periodic electrical interference from a corrupted MRI scan by locating and filtering it out in the frequency domain, then reconstructing a clean image.

## Implementation

I located two symmetric noise spikes in the k-space magnitude spectrum at (108, 109) and (148, 147), then built a notch mask with a radius-2 disc around each position (26 pixels total) and multiplied it directly with the frequency-domain data before reconstructing the image with the inverse Fourier transform.

## Results

The corrupted image showed a clear diagonal stripe pattern across the entire scan, while the filtered image removed this pattern completely and revealed the underlying anatomy; all 11 pytest checks passed, including SSIM comparisons confirming the filtered image closely matched the clean reference and no longer resembled the corrupted one.