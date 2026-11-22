# -*- coding: utf-8 -*-
"""Tests for Lab 06 — Scale Invariant Blob Detector.

Grouped by step — a failing test names the step that needs attention. Most
tests use small synthetic inputs so they stay fast; only the Step 10
end-to-end test builds a full scale space (a few seconds).
"""
import numpy as np
import math
import pytest
import cv2
from contextlib import redirect_stdout
from os import devnull
from scipy import signal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

with redirect_stdout(open(devnull, 'w')):
    from lab06 import ScaleInvDet

@pytest.fixture(scope="module")
def small_image():
    """A small 20x20 synthetic greyscale image — cheap for per-step tests."""
    rng = np.random.default_rng(0)
    return (rng.random((20, 20)) * 255).astype(np.float64)

def positions(sigma):
    rng = (int(6 * sigma + 1) - 1) // 2
    return np.arange(-rng, rng + 1)

def ref_gauss(sigma):
    x = positions(sigma)
    return (np.exp(-x ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma))[None, :]

def ref_gauss_2nd(sigma):
    x = positions(sigma)
    return (np.exp(-x ** 2 / (2 * sigma ** 2)) * (x ** 2 - sigma ** 2)
            / (np.sqrt(2 * np.pi) * sigma ** 5))[None, :]

def ref_laplacian(image, sigma):
    """Independent reference: two full 2D LoG-component kernels, by outer product."""
    G, Gxx = ref_gauss(sigma)[0], ref_gauss_2nd(sigma)[0]
    Lxx = signal.convolve2d(image, np.outer(G, Gxx), boundary='symm', mode='same')
    Lyy = signal.convolve2d(image, np.outer(Gxx, G), boundary='symm', mode='same')
    return Lxx, Lyy

def test_constructor_defaults():
    sid = ScaleInvDet()
    assert (sid.sigma_min, sid.sigma_max, sid.scale_fac, sid.thresh) == (1, 36, 1.05, 40)
    assert sid.num_scales == 33 and len(sid.sigma_vals) == 33

# ── Step 1: gauss_kernel ────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [2, 3.5])
def test_gauss_kernel_matches_formula(sigma):
    K = ScaleInvDet().gauss_kernel(sigma)
    assert K.shape == ref_gauss(sigma).shape
    assert np.allclose(K, ref_gauss(sigma))

# ── Step 2: gauss_2nd_deriv_kernel ──────────────────────────────────────────

@pytest.mark.parametrize("sigma", [2, 3.5])
def test_gauss_2nd_deriv_kernel_matches_formula(sigma):
    K = ScaleInvDet().gauss_2nd_deriv_kernel(sigma)
    assert K.shape == ref_gauss_2nd(sigma).shape
    assert np.allclose(K, ref_gauss_2nd(sigma))

def test_gauss_2nd_deriv_kernel_negative_at_centre():
    K = ScaleInvDet().gauss_2nd_deriv_kernel(2).ravel()
    assert K[len(K) // 2] < 0

# ── Step 3: second_derivatives ──────────────────────────────────────────────

def test_second_derivatives_match_reference(small_image):
    Lxx, Lyy = ScaleInvDet().second_derivatives(small_image, 2)
    ref_xx, ref_yy = ref_laplacian(small_image, 2)
    assert Lxx.shape == small_image.shape
    assert np.allclose(Lxx, ref_xx)
    assert np.allclose(Lyy, ref_yy)

def test_second_derivatives_directions():
    """Vertical stripes vary along x only, so Lxx is strong and Lyy is (almost) zero."""
    img = np.tile(np.sin(np.arange(30) / 2.0), (30, 1))
    Lxx, Lyy = ScaleInvDet().second_derivatives(img, 1.5)
    assert np.abs(Lyy).max() < 0.1 * np.abs(Lxx).max()

# ── Step 4: laplacian ───────────────────────────────────────────────────────

@pytest.mark.parametrize("sigma", [2, 3])
def test_laplacian_is_scale_normalised(small_image, sigma):
    ref_xx, ref_yy = ref_laplacian(small_image, sigma)
    assert np.allclose(ScaleInvDet().laplacian(small_image, sigma), sigma ** 2 * (ref_xx + ref_yy))

# ── Step 5: build_scale_space ───────────────────────────────────────────────

def test_build_scale_space_matches_per_scale_laplacian(small_image):
    sid = ScaleInvDet(sigma_min=1, sigma_max=3, scale_fac=0.5)
    scale_space = sid.build_scale_space(small_image)
    assert scale_space.shape == (*small_image.shape, sid.num_scales)
    for k, sigma in enumerate(sid.sigma_vals):
        assert np.allclose(scale_space[:, :, k], sid.laplacian(small_image, sigma))

# ── Step 6: candidate_points ────────────────────────────────────────────────

def test_candidate_points_above_threshold_and_interior():
    volume = np.zeros((6, 6, 4))
    volume[2, 3, 1] = 50      # interior, above 40          -> candidate
    volume[3, 3, 2] = 30      # interior, below 40          -> not
    volume[0, 2, 1] = 90      # first row (no neighbourhood) -> not
    volume[2, 2, 3] = 90      # last scale                  -> not
    result = ScaleInvDet().candidate_points(volume)
    assert result.shape == (1, 3)
    assert result.tolist() == [[2, 3, 1]]

def test_candidate_points_respects_thresh():
    volume = np.zeros((6, 6, 4))
    volume[2, 3, 1] = 50
    volume[3, 2, 2] = 30
    result = ScaleInvDet(thresh=20).candidate_points(volume)
    assert {tuple(p) for p in result} == {(2, 3, 1), (3, 2, 2)}

# ── Step 7: lap_max_det ─────────────────────────────────────────────────────

def demo_volume():
    volume = np.zeros((5, 5, 3))
    volume[2, 2, 1] = 100
    return volume

def test_lap_max_det_finds_true_maximum():
    sid = ScaleInvDet(sigma_min=1, sigma_max=4, scale_fac=1.0)   # sigma_vals = [1, 2.5, 4]
    result = sid.lap_max_det(demo_volume(), 2, 2, 1)
    assert len(result) == 4
    assert result[0] == 100 and result[1] == 2 and result[2] == 2 and result[3] == 2.5

@pytest.mark.parametrize("pos", [(1, 3, 1), (2, 2, 0), (2, 2, 2), (3, 1, 2)])
def test_lap_max_det_rejects_stronger_neighbour(pos):
    """A stronger point anywhere in the 3x3x3 block — including the scale above
    or below — means (row, col, k) is not a maximum."""
    volume = demo_volume()
    volume[2, 2, 1] = 50
    volume[pos] = 100
    assert ScaleInvDet().lap_max_det(volume, 2, 2, 1) == []

def test_lap_max_det_ignores_points_outside_neighbourhood():
    volume = demo_volume()
    volume[2, 4, 1] = 500      # two columns away
    assert ScaleInvDet().lap_max_det(volume, 2, 2, 1) != []

# ── Step 8: find_maxima ─────────────────────────────────────────────────────

def test_find_maxima_returns_every_maximum():
    sid = ScaleInvDet(sigma_min=1, sigma_max=4, scale_fac=1.0)    # 3 scales
    volume = np.zeros((8, 8, 3))
    volume[2, 2, 1] = 100
    volume[5, 5, 1] = 70
    volume[5, 6, 1] = 60      # next to (5, 5) -> not a maximum
    result = sid.find_maxima(volume)
    assert result.shape == (2, 4)
    assert {(int(r[1]), int(r[2])) for r in result} == {(2, 2), (5, 5)}

def test_find_maxima_empty():
    result = ScaleInvDet().find_maxima(np.zeros((6, 6, 3)))
    assert result.shape == (0, 4)

# ── Step 9: blob_circles ────────────────────────────────────────────────────

def test_blob_circles_radius_is_root2_sigma():
    maxima = np.array([[100.0, 10, 20, 2.0],
                       [ 50.0, 30, 40, 3.6]])
    result = ScaleInvDet().blob_circles(maxima)
    assert result.shape == (2, 3)
    assert np.issubdtype(result.dtype, np.integer)
    assert result.tolist() == [[10, 20, math.ceil(np.sqrt(2) * 2.0)],
                               [30, 40, math.ceil(np.sqrt(2) * 3.6)]]

# ── Step 10: detect_blobs (end-to-end) ──────────────────────────────────────

CIRCLES = [(50, 50, 5), (150, 50, 15), (50, 150, 30), (150, 150, 50), (100, 100, 75)]

@pytest.fixture(scope="module")
def circles_result():
    """A 200x200 image with 5 dark filled circles of known size and position."""
    test_image = np.ones((200, 200), dtype=np.uint8) * 255
    for x, y, d in CIRCLES:
        cv2.circle(test_image, (x, y), d // 2, (0,), -1)
    blurred = cv2.GaussianBlur(test_image, (19, 19), sigmaX=2)
    sid = ScaleInvDet()
    blobs = sid.detect_blobs(blurred)
    return sid, blobs

def test_detect_blobs_matches_known_circles(circles_result):
    """Each circle is matched by a blob within 10% of its diameter in position
    and 30% in radius."""
    _, blobs = circles_result
    assert blobs.ndim == 2 and blobs.shape[1] == 3
    matches = 0
    for cx, cy, d in CIRCLES:
        for row, col, radius in blobs:
            if np.hypot(cx - col, cy - row) < d * 0.1 and abs(radius - d // 2) < d * 0.3:
                matches += 1
                break
    assert matches == len(CIRCLES), f"matched {matches} of {len(CIRCLES)} circles"

def test_detect_blobs_stores_intermediates_on_self(circles_result):
    sid, blobs = circles_result
    assert sid.laplace_scale_space is not None and sid.laplace_scale_space.shape == (200, 200, 33)
    assert sid.maxima is not None and sid.maxima.shape[1] == 4
    assert np.array_equal(sid.blobs, blobs)

def test_detect_blobs_respects_thresh():
    """One dark disc (radius 4): found with a low threshold, not with a very high one."""
    img = np.full((40, 40), 255, np.uint8)
    cv2.circle(img, (20, 20), 4, (0,), -1)
    loose  = ScaleInvDet(sigma_max=6, scale_fac=0.5, thresh=10).detect_blobs(img)
    strict = ScaleInvDet(sigma_max=6, scale_fac=0.5, thresh=1e9).detect_blobs(img)
    assert len(loose) > 0 and len(strict) == 0
