# -*- coding: utf-8 -*-
"""Evaluate Lab 00 — Simple Image Processing.

Run from the repo root:
    python Assignments/lab00/evaluate_lab00.py

Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab00 import SimpleImageProcessing
from helpers.dataloader import get_data_path, load_image
from helpers.plotting import save_html

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

def _resize_for_display(image, max_width=640):
    """Downscale image to max_width keeping aspect ratio (avoids browser aliasing)."""
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)

def evaluate():
    # ── Load test images ────────────────────────────────────────────────────
    sphinx     = _resize_for_display(
        load_image(get_data_path("Great_Sphinx_of_Giza_-_20080716a.jpg"))
    )
    sacrecoeur = _resize_for_display(
        load_image(get_data_path("960px-Le_sacre_Coeur_bordercropped.jpg"))
    )

    # ── Instantiate the processor ────────────────────────────────────────────
    proc = SimpleImageProcessing()

    figs = []

    # ── Apply and plot filters ────────────────────────────────────────────────
    def _make_panel(image, label):
        blurred   = proc.add_blur(image, ksize=15)
        sharpened = proc.add_sharpen(image, strength=1.5)

        def _to_trace(img):
            return go.Image(z=img)

        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=["Original", "Blurred (ksize=15)", "Sharpened (strength=1.5)"],
        )
        for col, img in enumerate([image, blurred, sharpened], start=1):
            fig.add_trace(_to_trace(img), row=1, col=col)
        fig.update_layout(
            title_text=f"Lab 00: Simple Image Processing — {label}",
            height=380,
        )
        return fig

    figs.append(_make_panel(sphinx,     "Great Sphinx"))
    figs.append(_make_panel(sacrecoeur, "Sacré-Cœur"))

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab00_results.html"))

if __name__ == "__main__":
    evaluate()
