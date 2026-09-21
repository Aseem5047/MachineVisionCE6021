# -*- coding: utf-8 -*-
"""Evaluate Lab 00B — Gaussian Kernels, Convolution & Interpolation.

Run from the repo root:
    python Assignments/lab00B/evaluate_lab00B.py

Results are written to a single HTML file next to this script.
"""
import os
import sys
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab00B import ConvolutionInterpolator
from helpers.plotting import save_html

# ── Demo parameters ──────────────────────────────────────────────────────
# Test pattern: a "zone plate" / chirp — a 2D sinusoid whose local spatial
# frequency increases with distance from the centre (see _make_chirp_pattern
# below). Its frequency content sweeps continuously from low (centre) to
# just below the image's own Nyquist limit (edges/corners), which makes it
# a reliable, textbook way to expose aliasing: a real photograph is
# usually already band-limited by the camera and JPEG compression, so it
# won't show the effect nearly as clearly or consistently.
PATTERN_SIZE = 500
F0          = 10    # spatial frequency at the centre (R = 0)
CHIRP_RATE  = 250   # how fast the local frequency grows with R

SIGMA = 2.5
SCALE_FACTOR = 0.2
ORDER = 0   # nearest-neighbour: has no smoothing of its own, so any
            # aliasing (or lack of it) in the output comes only from
            # whether the input was pre-filtered before this call

def _make_chirp_pattern(size=PATTERN_SIZE, f0=F0, chirp_rate=CHIRP_RATE):
    """Generate a 2D chirp ("zone plate") test pattern.

    a(x, y) = sin(2*pi*(f0*R + chirp_rate*R**2 / 2)), where R = sqrt(x**2+y**2)
    over a square grid with x, y in [-0.5, 0.5). Its instantaneous local
    frequency grows linearly with R (from f0 at the centre), which is what
    makes it useful for exposing aliasing at any downsampling factor.
    """
    coords = np.arange(size) / size - 0.5
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X ** 2 + Y ** 2)
    return np.sin(np.pi * 2 * (f0 * R + chirp_rate * R ** 2 / 2))

def plot_aliasing_comparison(original, aliased, filtered):
    """Build a 3-panel Plotly figure comparing an original test pattern
    against a directly-interpolated ('aliased') and a Gaussian-prefiltered
    ('filtered') downsample of it.

    Each panel is a go.Heatmap with a greyscale colourscale — the same
    trace type used for frequency spectra in Lab 00C.
    """
    titles = ["Original pattern", "Direct interpolation (aliased)",
             "Gaussian-filtered interpolation"]
    panels = [original, aliased, filtered]

    fig = make_subplots(rows=1, cols=3, subplot_titles=titles)
    for col, panel in enumerate(panels, start=1):
        fig.add_trace(
            go.Heatmap(z=panel, colorscale="gray", showscale=False),
            row=1, col=col,
        )
    fig.update_yaxes(autorange="reversed")  # image rows run top -> bottom
    fig.update_layout(title_text="Lab 00B: Aliasing vs. Anti-Aliased Interpolation",
                      height=350)
    return fig

def run_aliasing_demo(pattern):
    """Run both the naive and Gaussian-prefiltered downsampling paths on
    pattern, using your ConvolutionInterpolator.

    Args:
        pattern (ndarray): H x W test pattern (e.g. from _make_chirp_pattern).

    Returns:
        aliased  (ndarray): pattern interpolated directly by SCALE_FACTOR,
                             with no pre-filtering.
        filtered (ndarray): pattern convolved with a Gaussian kernel
                             (sigma=SIGMA) first, then interpolated by the
                             same SCALE_FACTOR.
    """
    raise NotImplementedError("Implement this method")

def evaluate():
    print("Generating chirp test pattern...")
    pattern = _make_chirp_pattern()

    print("Running interpolation with and without a Gaussian pre-filter...")
    aliased, filtered = run_aliasing_demo(pattern)

    print("Building comparison figure...")
    fig = plot_aliasing_comparison(pattern, aliased, filtered)

    save_html(fig, output_path=os.path.join(LAB_DIR, "lab00B_results.html"))

if __name__ == "__main__":
    evaluate()
