# -*- coding: utf-8 -*-
"""Evaluate Lab 02 — MRI Noise Removal.

Run from the repo root:
    python Assignments/lab02/evaluate_lab02.py

Runs each step of MRIprocessor in turn and adds a figure for every step that
works. If a step isn't implemented yet, it stops there and still saves the
figures made so far — so you can run this after finishing each step.
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

from lab02 import MRIprocessor
from helpers.mridataprocessor import MRIDataProcessor
from helpers.dataloader import get_data_path
from helpers.plotting import save_html

HOVER = "row: %{y}<br>col: %{x}<br>value: %{z:.2f}<extra></extra>"

def plot_spectrum(fourier_img):
    """Step 1 figure: the Fourier magnitude, with (row, col) shown on hover."""
    fig = go.Figure(go.Heatmap(z=fourier_img, colorscale="Viridis", hovertemplate=HOVER))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(
        title_text="Step 1: Fourier magnitude — hover over the two bright "
                   "off-centre spikes to read their (row, col)",
        height=650, width=700,
    )
    return fig

def plot_image(img, title):
    """A single greyscale panel (used for steps 2 and 4)."""
    fig = go.Figure(go.Heatmap(z=img, colorscale="gray", showscale=False,
                               hovertemplate=HOVER))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(title_text=title, height=550, width=600)
    return fig

def plot_results(filtered_img, corrupted_img, fourier_img, mask):
    """Step 5 figure: a 2x2 grid of every stage of the pipeline."""
    panels = [
        (corrupted_img, "Corrupted image", "gray"),
        (fourier_img,   "Fourier magnitude", "Viridis"),
        (mask,          "Notch mask", "gray"),
        (filtered_img,  "Filtered image", "gray"),
    ]
    fig = make_subplots(rows=2, cols=2, subplot_titles=[p[1] for p in panels])
    for (img, _, scale), (r, c) in zip(panels, [(1, 1), (1, 2), (2, 1), (2, 2)]):
        fig.add_trace(go.Heatmap(z=img, colorscale=scale, showscale=False,
                                 hovertemplate=HOVER), row=r, col=c)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title_text="Step 5: Lab 02 MRI Noise Removal", height=800)
    return fig

def evaluate():
    print("Loading corrupted MRI k-space data...")
    k_space_1d = MRIDataProcessor.load_kspace_1d(get_data_path("corrupted_mrimage1d.txt"))
    f_shifted = k_space_1d.reshape((256, 256))

    processor = MRIprocessor(shape=(256, 256))
    figs = []
    step = ""

    try:
        step = "Step 1 (fourier_magnitude)"
        fourier_img = processor.fourier_magnitude(f_shifted)
        figs.append(plot_spectrum(fourier_img))
        print(f"{step} done.")

        step = "Step 2 (reconstruct)"
        corrupted_img = processor.reconstruct(f_shifted)
        figs.append(plot_image(corrupted_img, "Step 2: Reconstructed (corrupted) image"))
        print(f"{step} done.")

        step = "Step 3 (get_noise_peaks)"
        peaks = processor.get_noise_peaks()
        print(f"{step} done — noise peaks at (row, col): {peaks}")

        step = "Step 4 (make_notch_mask)"
        mask = processor.make_notch_mask(peaks)
        figs.append(plot_image(mask, "Step 4: Notch mask (black discs = removed frequencies)"))
        print(f"{step} done — {int((mask == 0).sum())} pixels notched out.")

        step = "Step 5 (remove_noise)"
        results = processor.remove_noise(f_shifted)
        figs.append(plot_results(*results))
        print(f"{step} done.")

    except NotImplementedError:
        print(f"\n{step} is not implemented yet — stopping here.")
        if figs:
            print("Figures for the earlier steps are still saved below.")

    if figs:
        save_html(*figs, output_path=os.path.join(LAB_DIR, "lab02_results.html"))

if __name__ == "__main__":
    evaluate()
