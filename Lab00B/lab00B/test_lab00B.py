# -*- coding: utf-8 -*-
"""Tests for Lab 00B — Gaussian Kernels, Convolution & Interpolation.

These tests recompute each expected result independently (rather than
comparing pixel-for-pixel to a saved reference image) and check structure
and numeric values — the same "no eyeballing required" approach used
throughout the course.
"""
import numpy as np
import os
import sys
import pytest
from scipy.signal import convolve2d
from skimage.transform import rescale

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lab00B import ConvolutionInterpolator

@pytest.fixture(scope="module")
def ci():
    return ConvolutionInterpolator()

# ── gaussian_kernel_2d ───────────────────────────────────────────────────

def test_gaussian_kernel_size_follows_rule_of_thumb(ci):
    sigma = 2.5
    expected_half = int(np.ceil(3 * sigma))   # +/- 3 sigma rule of thumb
    expected_K = 2 * expected_half + 1
    kernel = ci.gaussian_kernel_2d(sigma)
    assert kernel.shape == (expected_K, expected_K)

def test_gaussian_kernel_dtype(ci):
    kernel = ci.gaussian_kernel_2d(2.5)
    assert np.issubdtype(kernel.dtype, np.floating)

def test_gaussian_kernel_sums_to_one(ci):
    kernel = ci.gaussian_kernel_2d(1.5)
    assert np.isclose(kernel.sum(), 1.0)

def test_gaussian_kernel_matches_outer_product_reference(ci):
    sigma = 2.0
    half = int(np.ceil(3 * sigma))
    x = np.arange(-half, half + 1)
    g = (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-(x ** 2) / (2 * sigma ** 2))
    expected = np.outer(g, g)
    expected = expected / expected.sum()

    kernel = ci.gaussian_kernel_2d(sigma)
    assert np.allclose(kernel, expected)

def test_gaussian_kernel_peak_at_centre(ci):
    sigma = 1.5
    kernel = ci.gaussian_kernel_2d(sigma)
    centre = kernel.shape[0] // 2
    assert kernel[centre, centre] == kernel.max()

# ── convolve ─────────────────────────────────────────────────────────────

def test_convolve_matches_convolve2d_reference(ci):
    rng = np.random.default_rng(0)
    image  = rng.random((30, 30))
    kernel = ci.gaussian_kernel_2d(1.0)

    expected = convolve2d(image, kernel, mode='same', boundary='symm')
    result   = ci.convolve(image, kernel)
    assert np.allclose(result, expected)

def test_convolve_default_mode_preserves_shape(ci):
    image  = np.random.rand(25, 40)
    kernel = ci.gaussian_kernel_2d(1.0)
    result = ci.convolve(image, kernel)
    assert result.shape == image.shape

def test_convolve_respects_mode_kwarg(ci):
    image  = np.random.rand(20, 20)
    kernel = np.ones((5, 5)) / 25.0
    result = ci.convolve(image, kernel, mode='valid')
    assert result.shape == (16, 16)

def test_convolve_respects_boundary_kwarg(ci):
    image  = np.random.rand(20, 20)
    kernel = ci.gaussian_kernel_2d(1.0)

    expected = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    result   = ci.convolve(image, kernel, boundary='fill', fillvalue=0)
    assert np.allclose(result, expected)

def test_convolve_respects_fillvalue_kwarg(ci):
    # Uses a non-zero fillvalue so an implementation that hardcodes fillvalue=0
    # (its default) is caught rather than passing by coincidence.
    image  = np.random.rand(20, 20)
    kernel = ci.gaussian_kernel_2d(1.0)

    expected = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0.7)
    result   = ci.convolve(image, kernel, boundary='fill', fillvalue=0.7)
    assert np.allclose(result, expected)

def test_convolve_uniform_kernel_preserves_flat_region(ci):
    # A constant image should be unchanged by any normalised kernel.
    image  = np.full((15, 15), 0.5)
    kernel = ci.gaussian_kernel_2d(1.0)
    result = ci.convolve(image, kernel)
    assert np.allclose(result, 0.5, atol=1e-8)

# ── interpolate ──────────────────────────────────────────────────────────

def test_interpolate_output_shape(ci):
    image  = np.random.rand(40, 80)
    result = ci.interpolate(image, scale_factor=0.25)
    assert result.shape == (10, 20)

def test_interpolate_matches_rescale_reference(ci):
    rng = np.random.default_rng(1)
    image = rng.random((24, 36))
    result   = ci.interpolate(image, scale_factor=0.5, order=1)
    expected = rescale(image, 0.5, order=1, anti_aliasing=False, mode='reflect')
    assert np.allclose(result, expected)

def test_interpolate_default_scale_and_order(ci):
    image  = np.random.rand(16, 16)
    result = ci.interpolate(image)
    expected = rescale(image, 0.5, order=0, anti_aliasing=False, mode='reflect')
    assert result.shape == expected.shape
    assert np.allclose(result, expected)

def test_interpolate_has_no_implicit_anti_aliasing(ci):
    # If anti-aliasing were left on, the result would match skimage's own
    # anti-aliased rescale instead of the plain (no pre-filter) one.
    rng = np.random.default_rng(2)
    image = rng.random((30, 30))
    result = ci.interpolate(image, scale_factor=0.2, order=1)

    plain_ref    = rescale(image, 0.2, order=1, anti_aliasing=False, mode='reflect')
    smoothed_ref = rescale(image, 0.2, order=1, anti_aliasing=True,  mode='reflect')

    assert np.allclose(result, plain_ref)
    assert not np.allclose(result, smoothed_ref)
