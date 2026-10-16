# -*- coding: utf-8 -*-
"""Tests for Lab 02 — MRI Noise Removal via Fourier Notch Filtering.

One group of tests per step, so a failing test tells you which step to fix.
"""
import pytest
import numpy as np
import os
import sys
from skimage.metrics import structural_similarity as ssim
from skimage.io import imread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from lab02 import MRIprocessor
from helpers.mridataprocessor import MRIDataProcessor
from helpers.dataloader import get_data_path

SHAPE = (256, 256)
TRUE_PEAKS = [(148, 147), (108, 109)]   # (row, col) of the two noise spikes
PEAK_TOLERANCE = 2                      # pixels

THRESHOLD_FILTERED  = 0.85  # filtered output must match reference
THRESHOLD_CORRUPTED = 0.50  # filtered output must NOT resemble corrupted reference

@pytest.fixture(scope="module")
def f_shifted():
    k_space_1d = MRIDataProcessor.load_kspace_1d(get_data_path("corrupted_mrimage1d.txt"))
    return k_space_1d.reshape(SHAPE)

@pytest.fixture(scope="module")
def processor():
    return MRIprocessor(shape=SHAPE)

def _normalise_to_uint8(img):
    lo, hi = img.min(), img.max()
    if hi - lo == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return (255 * (img - lo) / (hi - lo)).astype(np.uint8)

def _load_ref(name):
    img = imread(get_data_path(name), as_gray=True)
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    return img

# ── Step 1: fourier_magnitude ────────────────────────────────────────────

def test_fourier_magnitude_values(processor, f_shifted):
    result = processor.fourier_magnitude(f_shifted)
    assert result.shape == SHAPE
    assert np.isrealobj(result), "Magnitude must be real, not complex."
    assert np.allclose(result, np.log(np.abs(f_shifted) + 1))

# ── Step 2: reconstruct ──────────────────────────────────────────────────

def test_reconstruct_shape_and_type(processor, f_shifted):
    img = processor.reconstruct(f_shifted)
    assert img.shape == SHAPE
    assert np.isrealobj(img) and np.issubdtype(img.dtype, np.floating), (
        f"Expected a real float image, got dtype {img.dtype}."
    )

def test_reconstruct_matches_corrupted_reference(processor, f_shifted):
    """The unfiltered reconstruction should look like the corrupted reference.

    A low score usually means the image is not centred — check you applied
    np.fft.fftshift to the result of the inverse FFT.
    """
    img = _normalise_to_uint8(processor.reconstruct(f_shifted))
    score = ssim(_load_ref("corrupted_image.png"), img, data_range=255)
    assert score >= 0.95, f"SSIM vs corrupted reference was {score:.4f}."

# ── Step 3: get_noise_peaks ──────────────────────────────────────────────

def test_noise_peaks_found(processor):
    peaks = processor.get_noise_peaks()
    assert len(peaks) == 2, f"Expected 2 peaks, got {len(peaks)}."
    for true_r, true_c in TRUE_PEAKS:
        nearest = min(np.hypot(r - true_r, c - true_c) for r, c in peaks)
        assert nearest <= PEAK_TOLERANCE, (
            f"No peak within {PEAK_TOLERANCE} px of a noise spike — "
            f"re-check the spectrum plot. Remember positions are (row, col)."
        )

# ── Step 4: make_notch_mask ──────────────────────────────────────────────

def test_mask_shape_and_values(processor):
    mask = processor.make_notch_mask([(100, 100)], radius=3)
    assert mask.shape == SHAPE
    assert set(np.unique(mask)).issubset({0, 1}), "Mask values must be 0 or 1."

def test_mask_zero_at_centre_one_elsewhere(processor):
    mask = processor.make_notch_mask([(100, 120)], radius=3)
    assert mask[100, 120] == 0, "The notch centre must be zero (remember (row, col))."
    assert mask[0, 0] == 1 and mask[255, 255] == 1, "Pixels far from a notch must be one."

def test_mask_notches_every_centre(processor):
    centres = [(50, 60), (200, 180)]
    mask = processor.make_notch_mask(centres, radius=2)
    for r, c in centres:
        assert mask[r, c] == 0, f"Centre ({r}, {c}) was not notched out."

def test_mask_disc_area(processor):
    radius = 5
    mask = processor.make_notch_mask([(128, 128)], radius=radius)
    zeros = int((mask == 0).sum())
    expected = np.pi * radius ** 2
    assert 0.7 * expected <= zeros <= 1.3 * expected, (
        f"Expected roughly {expected:.0f} zeroed pixels for a radius-{radius} disc, got {zeros}."
    )

def test_mask_respects_radius(processor):
    small = processor.make_notch_mask([(128, 128)], radius=2)
    large = processor.make_notch_mask([(128, 128)], radius=6)
    assert (large == 0).sum() > (small == 0).sum(), "A larger radius must notch out more pixels."

# ── Step 5: remove_noise ─────────────────────────────────────────────────

def test_remove_noise_returns_four_outputs(processor, f_shifted):
    results = processor.remove_noise(f_shifted)
    assert isinstance(results, tuple) and len(results) == 4, (
        "remove_noise must return (filtered_img, corrupted_img, fourier_img, mask)."
    )
    filtered_img = results[0]
    assert filtered_img.shape == SHAPE
    assert np.issubdtype(filtered_img.dtype, np.floating)

def test_remove_noise_similarity(processor, f_shifted):
    """Filtered image must closely match reference and differ from corrupted reference."""
    filtered_img = processor.remove_noise(f_shifted)[0]
    test_img = _normalise_to_uint8(filtered_img)

    ssim_vs_corrupted = ssim(_load_ref("corrupted_image.png"), test_img, data_range=255)
    ssim_vs_filtered  = ssim(_load_ref("filtered_image.png"),  test_img, data_range=255)
    print(f"\nSSIM vs corrupted reference: {ssim_vs_corrupted:.4f}")
    print(f"SSIM vs filtered  reference: {ssim_vs_filtered:.4f}")

    assert ssim_vs_corrupted < THRESHOLD_CORRUPTED, (
        f"Noise not removed: output too similar to corrupted reference "
        f"(SSIM {ssim_vs_corrupted:.4f} >= {THRESHOLD_CORRUPTED})."
    )
    assert ssim_vs_filtered >= THRESHOLD_FILTERED, (
        f"Filtered image SSIM ({ssim_vs_filtered:.4f}) below threshold {THRESHOLD_FILTERED}."
    )
