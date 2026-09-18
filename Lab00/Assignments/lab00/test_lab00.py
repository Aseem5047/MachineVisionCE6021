# -*- coding: utf-8 -*-
"""Tests for Lab 00 — Simple Image Processing.

Verifies blur and sharpen using frequency-domain analysis:
  - Blur  should reduce high-frequency energy relative to the original.
  - Sharpen should reduce low-frequency energy relative to the original
    (i.e. the high-to-low frequency ratio increases after sharpening).
"""
import numpy as np
import cv2
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lab00 import SimpleImageProcessing
from helpers.dataloader import get_data_path, load_image

def _freq_energy(image):
    """Return (low_energy, high_energy) split at the median frequency.

    The image is converted to greyscale, FFT applied, then the magnitude
    spectrum is split at the midpoint radius into low and high bands.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    F    = np.fft.fft2(gray.astype(np.float32))
    Fsh  = np.fft.fftshift(F)
    mag  = np.abs(Fsh)

    H, W = gray.shape
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    r_max = min(cy, cx)

    low_mask  = dist <= r_max / 2
    high_mask = dist > r_max / 2

    return float(mag[low_mask].sum()), float(mag[high_mask].sum())

@pytest.fixture(scope="module")
def images_and_proc():
    sphinx     = load_image(get_data_path("Great_Sphinx_of_Giza_-_20080716a.jpg"))
    sacrecoeur = load_image(get_data_path("960px-Le_sacre_Coeur_bordercropped.jpg"))
    proc       = SimpleImageProcessing()
    return sphinx, sacrecoeur, proc

@pytest.mark.parametrize("img_key", ["sphinx", "sacrecoeur"])
def test_blur_reduces_high_frequency(img_key, images_and_proc):
    """Blurring must reduce the high-frequency energy of the image."""
    sphinx, sacrecoeur, proc = images_and_proc
    image = sphinx if img_key == "sphinx" else sacrecoeur

    blurred = proc.add_blur(image, ksize=21)

    _, hf_orig    = _freq_energy(image)
    _, hf_blurred = _freq_energy(blurred)

    assert hf_blurred < hf_orig, (
        f"[{img_key}] Expected blur to reduce high-frequency energy, "
        f"but got {hf_blurred:.1f} ≥ {hf_orig:.1f}"
    )

@pytest.mark.parametrize("img_key", ["sphinx", "sacrecoeur"])
def test_sharpen_increases_hf_to_lf_ratio(img_key, images_and_proc):
    """Sharpening must boost high frequencies relative to low frequencies."""
    sphinx, sacrecoeur, proc = images_and_proc
    image = sphinx if img_key == "sphinx" else sacrecoeur

    sharpened = proc.add_sharpen(image, strength=1.5)

    lf_orig, hf_orig      = _freq_energy(image)
    lf_sharp, hf_sharp    = _freq_energy(sharpened)

    ratio_orig  = hf_orig  / lf_orig  if lf_orig  > 0 else 0
    ratio_sharp = hf_sharp / lf_sharp if lf_sharp > 0 else 0

    assert ratio_sharp > ratio_orig, (
        f"[{img_key}] Expected sharpen to increase HF/LF ratio, "
        f"but got {ratio_sharp:.4f} ≤ {ratio_orig:.4f}"
    )

def test_blur_output_shape_and_dtype(images_and_proc):
    """Blurred output must match the input shape and be uint8."""
    sphinx, _, proc = images_and_proc
    blurred = proc.add_blur(sphinx, ksize=15)
    assert blurred.shape == sphinx.shape
    assert blurred.dtype == np.uint8

def test_sharpen_output_shape_and_dtype(images_and_proc):
    """Sharpened output must match the input shape and be uint8."""
    sphinx, _, proc = images_and_proc
    sharpened = proc.add_sharpen(sphinx, strength=1.5)
    assert sharpened.shape == sphinx.shape
    assert sharpened.dtype == np.uint8

def test_blur_respects_ksize(images_and_proc):
    """A larger ksize must blur more strongly (remove more high-frequency energy).

    Guards against an implementation that ignores the ksize kwarg and
    hardcodes a fixed kernel size.
    """
    sphinx, _, proc = images_and_proc
    small_k = proc.add_blur(sphinx, ksize=3)
    large_k = proc.add_blur(sphinx, ksize=31)

    _, hf_small = _freq_energy(small_k)
    _, hf_large = _freq_energy(large_k)

    assert hf_large < hf_small, (
        f"Expected a larger ksize to reduce high-frequency energy further "
        f"(ksize=31 gave {hf_large:.1f}, ksize=3 gave {hf_small:.1f}) — "
        f"is the ksize kwarg actually being used?"
    )

def test_sharpen_respects_strength(images_and_proc):
    """A larger strength must sharpen more (raise the HF/LF energy ratio further).

    Guards against an implementation that ignores the strength kwarg.
    """
    sphinx, _, proc = images_and_proc
    weak   = proc.add_sharpen(sphinx, strength=0.5)
    strong = proc.add_sharpen(sphinx, strength=3.0)

    lf_w, hf_w = _freq_energy(weak)
    lf_s, hf_s = _freq_energy(strong)

    ratio_weak   = hf_w / lf_w if lf_w > 0 else 0
    ratio_strong = hf_s / lf_s if lf_s > 0 else 0

    assert ratio_strong > ratio_weak, (
        f"Expected a larger strength to increase the HF/LF ratio further "
        f"({ratio_strong:.4f} vs {ratio_weak:.4f}) — "
        f"is the strength kwarg actually being used?"
    )
