# -*- coding: utf-8 -*-
"""Evaluate Lab 00C — Visualizing Results with Plotly.

Run from the repo root:
    python Assignments/lab00C/evaluate_lab00C.py

Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab00C import ResultsVisualizer
from helpers.dataloader import get_data_path, load_image
from helpers.plotting import save_html

def _resize_for_display(image, max_width=640):
    """Downscale image to max_width keeping aspect ratio (avoids browser aliasing)."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def _fft_magnitude(image):
    """Return the log-magnitude FFT spectrum of a (greyscale) image."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    return np.log1p(np.abs(F))

def _row_profile(image, row_frac=0.5):
    """Return the greyscale intensity profile along one horizontal row."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    row = int(gray.shape[0] * row_frac)
    return gray[row, :]

def evaluate():
    # ── Load test images (same pair used in Lab 00) ─────────────────────────
    sphinx = _resize_for_display(
        load_image(get_data_path("Great_Sphinx_of_Giza_-_20080716a.jpg"))
    )
    sacrecoeur = _resize_for_display(
        load_image(get_data_path("960px-Le_sacre_Coeur_bordercropped.jpg"))
    )

    viz = ResultsVisualizer()
    figs = []

    # ── 1. Image grid ────────────────────────────────────────────────────────
    print("Building image grid...")
    figs.append(viz.image_grid(
        [sphinx, sacrecoeur],
        titles=["Great Sphinx", "Sacré-Cœur"],
        suptitle="Lab 00C: Test Images",
    ))

    # ── 2. Heatmap — FFT log-magnitude spectrum ─────────────────────────────
    print("Building heatmap...")
    figs.append(viz.heatmap(
        _fft_magnitude(sphinx),
        title="Lab 00C: Fourier Magnitude Spectrum (Great Sphinx)",
        colorscale="Viridis",
    ))

    # ── 3. Line chart — horizontal intensity profile ────────────────────────
    print("Building line chart...")
    profile_sphinx = _row_profile(sphinx)
    profile_sacrecoeur = _row_profile(sacrecoeur)
    x = np.arange(len(profile_sphinx))
    figs.append(viz.line_chart(
        x,
        {
            "Great Sphinx": profile_sphinx,
            "Sacré-Cœur":   profile_sacrecoeur,
        },
        title="Lab 00C: Horizontal Intensity Profile (middle row)",
        xaxis_title="Pixel column",
        yaxis_title="Greyscale intensity",
    ))

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab00C_results.html"))

if __name__ == "__main__":
    evaluate()
