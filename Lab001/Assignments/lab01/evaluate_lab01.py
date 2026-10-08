# -*- coding: utf-8 -*-
"""Evaluate Lab 01 — Gaussian Filter Convolution.

Run from the repo root:
    python Assignments/lab01/evaluate_lab01.py

Builds a side-by-side plotly figure of the original greyscale image and the
result after Gaussian filtering, prints the convolution timing, and writes
the figure to a single HTML file next to this script.
"""
import time
import sys
import os
import numpy as np

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab01 import GaussianFilt, read_image
from helpers.plotting import save_html

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def evaluate():
    print("Loading image...")
    image_scale, gray = read_image()

    sigma = 10
    my_gauss_filt = GaussianFilt(sigma=sigma)

    print(f"Running Gaussian convolution (sigma={sigma})...")
    start_time = time.time()
    gauss_conv_img = my_gauss_filt.my_conv_method(gray)
    elapsed = time.time() - start_time

    print(f"Time taken to compute convolution: {elapsed:.4f} seconds")

    # Build side-by-side figure using plotly
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Original Greyscale", f"Gaussian Filtered (σ={sigma})"],
    )
    fig.add_trace(go.Heatmap(z=gray, colorscale="gray", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=gauss_conv_img, colorscale="gray", showscale=False), row=1, col=2)
    fig.update_yaxes(autorange="reversed")  # image rows run top -> bottom
    fig.update_layout(title_text="Lab 01: Gaussian Convolution", height=500)

    save_html(fig, output_path=os.path.join(LAB_DIR, "lab01_results.html"))

if __name__ == "__main__":
    evaluate()
