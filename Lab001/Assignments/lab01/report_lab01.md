# Lab Report — Lab 01: Efficient Separable Gaussian Convolution

## Overview

The goal was to blur an image with a Gaussian filter, but do it in a fast way by splitting one slow 2D blur into two quick 1D blurs (one across rows, one across columns) instead.

## Implementation

I built a small 1D Gaussian kernel from sigma, padded the image on all sides so the output stays the same size, then blurred it first along the rows and then along the columns using that same kernel. Instead of looping over pixels one by one in Python (which is slow), I used NumPy to do the blurring on the whole image at once.

## Results

My version ran about 50–80 times faster than the slow, direct 2D blur method, easily passing all five speed checks, including the strictest one.