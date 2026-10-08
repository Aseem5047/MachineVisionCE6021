# -*- coding: utf-8 -*-
import pytest
import time
import numpy as np
from scipy.signal import convolve2d

from lab01 import GaussianFilt, read_image

# Grading thresholds, expressed as a fraction of a naive-baseline time rather
# than an absolute number of seconds. The baseline is a direct 2D convolve2d
# using the full (outer-product) Gaussian kernel — the "obviously correct but
# not optimised" method this lab asks you to beat. Scale 1.0 means "at least
# as fast as the naive method"; smaller scales are progressively stricter.
SCALED_BASELINE = [1.0, 0.75, 0.5, 0.25, 0.1]

@pytest.fixture(scope="module")
def conv_execution():
    """Run once per session: load image, execute convolution, record timing,
    and time a naive-baseline convolve2d for comparison."""
    image_scale, gray = read_image()
    my_gauss_filt = GaussianFilt(sigma=10)

    start_time = time.time()
    result_img = my_gauss_filt.my_conv_method(gray)
    run_time = time.time() - start_time

    # Baseline: direct 2D convolution with the full (outer-product) kernel.
    kernel_1d = my_gauss_filt.gauss_kernel()
    kernel_2d = kernel_1d @ kernel_1d.T

    baseline_start = time.time()
    convolve2d(gray, kernel_2d, mode="same")
    baseline_time = time.time() - baseline_start

    print(f"\n[Fixture] Baseline (convolve2d, same padding): {baseline_time:.4f}s")
    print(f"[Fixture] Your implementation: {run_time:.4f}s "
          f"({run_time / baseline_time:.2f}x baseline)")
    return {
        "run_time": run_time,
        "baseline_time": baseline_time,
        "result_img": result_img,
        "original_shape": gray.shape,
    }

@pytest.mark.parametrize("scale", SCALED_BASELINE)
def test_execution_time(conv_execution, scale):
    """Check that execution time is within `scale` x the convolve2d baseline time."""
    actual_time = conv_execution["run_time"]
    threshold = conv_execution["baseline_time"] * scale
    assert actual_time <= threshold, (
        f"Performance Check Failed: took {actual_time:.4f}s, "
        f"exceeds {scale:.2f}x baseline threshold of {threshold:.4f}s "
        f"(baseline convolve2d took {conv_execution['baseline_time']:.4f}s)."
    )

def test_output_shape(conv_execution):
    """Output image must have the same spatial dimensions as the input."""
    assert conv_execution["result_img"].shape == conv_execution["original_shape"], (
        f"Shape mismatch: expected {conv_execution['original_shape']}, "
        f"got {conv_execution['result_img'].shape}."
    )

def test_output_type(conv_execution):
    """Output must be a uint8 numpy array."""
    result_img = conv_execution["result_img"]
    assert isinstance(result_img, np.ndarray), "Output must be a numpy array."
    assert result_img.dtype == np.uint8, (
        f"Expected dtype uint8, got {result_img.dtype}."
    )

@pytest.mark.parametrize("sigma", [2, 5, 10])
def test_gauss_kernel_is_1d_of_expected_size(sigma):
    """gauss_kernel() must return a 1D kernel of length int(6 * sigma + 1)."""
    my_gauss_filt = GaussianFilt(sigma=sigma)
    kernel = my_gauss_filt.gauss_kernel()
    expected_size = int(6 * sigma + 1)

    assert kernel.shape == (expected_size, 1), (
        f"Expected shape ({expected_size}, 1), got {kernel.shape}."
    )
    assert kernel.squeeze().ndim == 1, "Kernel must be 1D once squeezed to a vector."
