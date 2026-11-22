# -*- coding: utf-8 -*-
"""Evaluate Lab 06 — Scale Invariant Blob Detector.

Run from the repo root:
    python Assignments/lab06/evaluate_lab06.py

Runs each step of ScaleInvDet in turn and adds a figure for every step that
works. If a step isn't implemented yet, it stops there and still saves the
figures made so far — so you can run this after finishing each step.

Note: Steps 5 and 10 each build the full scale space on the sunflowers image,
which takes around 20-30 seconds.

Results are written to a single HTML file next to this script.
"""
import os
import sys
import cv2
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

LAB_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, LAB_DIR)
sys.path.insert(0, os.path.join(LAB_DIR, ".."))

from lab06 import ScaleInvDet, FLOWERS_IMAGE_URL
from helpers.dataloader import DATA_DIR, download_image, norm_img
from helpers.plotting import save_html, png_source, jpeg_source

HOVER = "x: %{x}<br>y: %{y}<extra></extra>"
DISPLAY_MAX_WIDTH = 700   # panels are shrunk before embedding; the algorithm
                          # itself always runs at full resolution
DEMO_SIGMA = 4            # a single representative scale for Steps 1-4

def resize_for_display(img, max_width=DISPLAY_MAX_WIDTH):
    """Downscale an image (keeping aspect ratio) purely for display."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    return cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)

def to_display(arr):
    """A response map as an RGB heat map (jet colour map: blue = weak,
    red = strong), squared so that sign doesn't matter."""
    grey = (norm_img(arr.astype(np.float64) ** 2) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(grey, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)

def plot_image(img, title, photo=False):
    """A single-panel image figure. Photographs as JPEG, response maps as PNG."""
    small = resize_for_display(img)
    source = jpeg_source(small) if photo else png_source(small)
    fig = go.Figure(go.Image(source=source, hovertemplate=HOVER))
    fig.update_layout(title_text=title, height=560)
    return fig

def plot_panels(panels, title):
    """panels: list of (array, subtitle) pairs, shown side by side as images."""
    fig = make_subplots(rows=1, cols=len(panels), subplot_titles=[p[1] for p in panels])
    for col, (arr, _) in enumerate(panels, start=1):
        fig.add_trace(go.Image(source=png_source(resize_for_display(arr, 500)),
                               hovertemplate=HOVER), row=1, col=col)
    fig.update_layout(title_text=title, height=420)
    return fig

def plot_kernels(kernels, title):
    """kernels: list of (1 x k kernel, name) pairs, drawn as line plots."""
    fig = go.Figure()
    for kernel, name in kernels:
        values = kernel.ravel()
        x = np.arange(len(values)) - len(values) // 2
        fig.add_trace(go.Scatter(x=x, y=values, mode="lines+markers", name=name))
    fig.update_layout(title_text=title, height=380, xaxis_title="position (px)")
    return fig

def plot_scale_space_grid(scale_space, sigma_vals, title, n_display=12):
    """A grid of representative scales from the scale space."""
    indices = np.linspace(0, scale_space.shape[2] - 1, min(n_display, scale_space.shape[2]), dtype=int)
    rows, cols = 3, 4
    fig = make_subplots(rows=rows, cols=cols,
                        subplot_titles=[f"sigma={sigma_vals[i]:.1f}" for i in indices])
    for pos, idx in enumerate(indices):
        display = resize_for_display(to_display(scale_space[:, :, idx]), 260)
        fig.add_trace(go.Image(source=png_source(display)), row=pos // cols + 1, col=pos % cols + 1)
    fig.update_layout(title_text=title, height=700)
    return fig

def plot_candidates_per_scale(candidates, sigma_vals, title):
    counts = np.bincount(candidates[:, 2], minlength=len(sigma_vals)) if len(candidates) else \
             np.zeros(len(sigma_vals), int)
    fig = go.Figure(go.Bar(x=np.round(sigma_vals, 1), y=counts,
                           hovertemplate="sigma: %{x}<br>candidates: %{y}<extra></extra>"))
    fig.update_layout(title_text=title, height=380, xaxis_title="sigma",
                      yaxis_title="number of candidates")
    return fig

def plot_neighbourhood(volume, title):
    """The three scale layers of a small synthetic scale space, as heatmaps."""
    fig = make_subplots(rows=1, cols=3, subplot_titles=["k = 0", "k = 1", "k = 2"])
    for k in range(3):
        fig.add_trace(go.Heatmap(z=volume[:, :, k], zmin=0, zmax=volume.max(),
                                 colorscale="Viridis", showscale=(k == 2),
                                 hovertemplate="row: %{y}<br>col: %{x}<br>value: %{z}<extra></extra>"),
                      row=1, col=k + 1)
        # Square cells (otherwise the heatmap is stretched to fill the panel).
        fig.update_xaxes(constrain="domain", row=1, col=k + 1)
        fig.update_yaxes(autorange="reversed", scaleanchor=f"x{k + 1 if k > 0 else ''}",
                         constrain="domain", row=1, col=k + 1)
    fig.update_layout(title_text=title, height=360)
    return fig

def draw_points(image_rgb, maxima):
    out = image_rgb.copy()
    for _, row, col, _ in maxima:
        cv2.circle(out, (int(col), int(row)), 3, (255, 0, 0), -1)
    return out

def draw_circles(image_rgb, blobs):
    out = image_rgb.copy()
    for row, col, radius in blobs:
        cv2.circle(out, (int(col), int(row)), int(radius), (255, 0, 0), 2)
    return out

def get_flowers_image():
    image_rgb = download_image(FLOWERS_IMAGE_URL, save_path=DATA_DIR)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    return image_rgb, gray

def evaluate():
    image_rgb, gray = get_flowers_image()

    sid = ScaleInvDet()
    figs = [plot_image(image_rgb, "Input image — sunflowers", photo=True)]
    step = ""

    try:
        step = "Step 1 (gauss_kernel)"
        G = sid.gauss_kernel(DEMO_SIGMA)
        figs.append(plot_kernels([(G, "Gaussian")],
                                 f"Step 1: 1D Gaussian kernel (sigma={DEMO_SIGMA}, {G.shape[1]} values) "
                                 "— a single smooth bump"))
        print(f"{step} done — kernel shape {G.shape}.")

        step = "Step 2 (gauss_2nd_deriv_kernel)"
        Gxx = sid.gauss_2nd_deriv_kernel(DEMO_SIGMA)
        figs.append(plot_kernels([(Gxx, "2nd derivative")],
                                 f"Step 2: Second-derivative kernel (sigma={DEMO_SIGMA}) — negative "
                                 "in the middle, crossing zero at ±sigma"))
        print(f"{step} done — kernel shape {Gxx.shape}.")

        step = "Step 3 (second_derivatives)"
        Lxx, Lyy = sid.second_derivatives(gray, DEMO_SIGMA)
        figs.append(plot_panels([(to_display(Lxx), "Lxx"), (to_display(Lyy), "Lyy")],
                                f"Step 3: Second derivatives (sigma={DEMO_SIGMA}, squared) — Lxx "
                                "picks out vertical structure, Lyy horizontal"))
        print(f"{step} done.")

        step = "Step 4 (laplacian)"
        response = sid.laplacian(gray, DEMO_SIGMA)
        figs.append(plot_image(to_display(response),
                               f"Step 4: Scale-normalised LoG (sigma={DEMO_SIGMA}, squared) — "
                               "strongest on blobs about this size"))
        print(f"{step} done.")

        step = "Step 5 (build_scale_space)"
        scale_space = sid.build_scale_space(gray)
        figs.append(plot_scale_space_grid(
            scale_space, sid.sigma_vals,
            f"Step 5: Scale space ({sid.num_scales} scales, 12 shown) — small blobs respond "
            "at small sigma, large blobs at large sigma"))
        print(f"{step} done — shape {scale_space.shape}.")

        step = "Step 6 (candidate_points)"
        candidates = sid.candidate_points(scale_space)
        figs.append(plot_candidates_per_scale(
            candidates, sid.sigma_vals,
            f"Step 6: {len(candidates)} candidate points above thresh={sid.thresh}, per scale"))
        print(f"{step} done — {len(candidates)} candidates.")

        step = "Step 7 (lap_max_det)"
        # A tiny hand-built scale space: a single peak at (2, 2, 1), and a
        # weaker point at (1, 3, 1) next to it.
        demo = np.zeros((5, 5, 3))
        demo[2, 2, 1] = 100
        demo[1, 3, 1] = 60
        peak  = sid.lap_max_det(demo, 2, 2, 1)
        other = sid.lap_max_det(demo, 1, 3, 1)
        figs.append(plot_neighbourhood(
            demo, f"Step 7: lap_max_det at (2, 2, 1) → {'maximum' if peak else '[]'}; "
                  f"at (1, 3, 1) → {'maximum' if other else '[] (stronger neighbour)'}"))
        print(f"{step} done — (2, 2, 1) gives {peak}; (1, 3, 1) gives {other}.")

        step = "Step 8 (find_maxima)"
        maxima = sid.find_maxima(scale_space)
        figs.append(plot_image(draw_points(image_rgb, maxima),
                               f"Step 8: {len(maxima)} local maxima in (row, col, scale) "
                               "— most sit on flower centres", photo=True))
        print(f"{step} done — {len(maxima)} maxima.")

        step = "Step 9 (blob_circles)"
        blobs = sid.blob_circles(maxima)
        figs.append(plot_image(draw_circles(image_rgb, blobs),
                               "Step 9: One circle per maximum, radius = √2 · sigma — "
                               "larger circles on the larger (nearer) flowers", photo=True))
        print(f"{step} done — radii from {blobs[:, 2].min()} to {blobs[:, 2].max()} px."
              if len(blobs) else f"{step} done — no blobs.")

        step = "Step 10 (detect_blobs)"
        blobs = sid.detect_blobs(gray)
        figs.append(plot_image(draw_circles(image_rgb, blobs),
                               f"Step 10: Full pipeline — {len(blobs)} blobs detected", photo=True))
        print(f"{step} done — {len(blobs)} blobs detected.")

    except NotImplementedError:
        print(f"\n{step} is not implemented yet — stopping here.")
        print("Figures for the earlier steps are still saved below.")

    save_html(*figs, output_path=os.path.join(LAB_DIR, "lab06_results.html"))

if __name__ == "__main__":
    evaluate()
