# -*- coding: utf-8 -*-
"""Tests for Lab 00C — Visualizing Results with Plotly.

These tests check the *structure* of the returned Plotly figures (trace
count, trace type, data values) rather than rendering pixels — the same
"no eyeballing required" approach used throughout the course.
"""
import numpy as np
import os
import sys
import pytest
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from lab00C import ResultsVisualizer

@pytest.fixture(scope="module")
def viz():
    return ResultsVisualizer()

# ── image_grid ───────────────────────────────────────────────────────────

def test_image_grid_returns_figure(viz):
    images = [np.zeros((8, 8, 3), dtype=np.uint8), np.full((8, 8, 3), 255, dtype=np.uint8)]
    fig = viz.image_grid(images, titles=["A", "B"])
    assert isinstance(fig, go.Figure)

def test_image_grid_one_trace_per_image(viz):
    images = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]
    fig = viz.image_grid(images, titles=["1", "2", "3"])
    assert len(fig.data) == 3
    assert all(trace.type == "image" for trace in fig.data)

def test_image_grid_default_titles(viz):
    images = [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8)]
    fig = viz.image_grid(images)
    annotation_text = [a.text for a in fig.layout.annotations]
    assert "Image 1" in annotation_text
    assert "Image 2" in annotation_text

# ── heatmap ──────────────────────────────────────────────────────────────

def test_heatmap_returns_figure_with_heatmap_trace(viz):
    matrix = np.random.rand(10, 10)
    fig = viz.heatmap(matrix, title="Test Heatmap")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "heatmap"

def test_heatmap_preserves_values(viz):
    matrix = np.arange(9).reshape(3, 3).astype(float)
    fig = viz.heatmap(matrix)
    assert np.array_equal(np.array(fig.data[0].z), matrix)

def test_heatmap_title_applied(viz):
    fig = viz.heatmap(np.zeros((3, 3)), title="My Title")
    assert fig.layout.title.text == "My Title"

def test_heatmap_respects_colorscale(viz):
    """The colorscale kwarg must reach the trace, not be hardcoded to the default."""
    fig_default = viz.heatmap(np.zeros((3, 3)))
    fig_custom  = viz.heatmap(np.zeros((3, 3)), colorscale="Cividis")
    assert fig_custom.data[0].colorscale != fig_default.data[0].colorscale

# ── line_chart ───────────────────────────────────────────────────────────

def test_line_chart_one_trace_per_series(viz):
    x = np.arange(5)
    series = {"a": np.arange(5), "b": np.arange(5) * 2}
    fig = viz.line_chart(x, series)
    assert len(fig.data) == 2
    assert all(trace.type == "scatter" for trace in fig.data)

def test_line_chart_trace_names_match_series_keys(viz):
    x = np.arange(5)
    series = {"first": np.arange(5), "second": np.arange(5)}
    fig = viz.line_chart(x, series)
    names = {trace.name for trace in fig.data}
    assert names == {"first", "second"}

def test_line_chart_axis_titles(viz):
    x = np.arange(3)
    fig = viz.line_chart(x, {"s": np.arange(3)},
                          xaxis_title="Time", yaxis_title="Value")
    assert fig.layout.xaxis.title.text == "Time"
    assert fig.layout.yaxis.title.text == "Value"

def test_line_chart_values_correct(viz):
    x = np.array([0, 1, 2])
    y = np.array([10, 20, 30])
    fig = viz.line_chart(x, {"s": y})
    assert np.array_equal(np.array(fig.data[0].x), x)
    assert np.array_equal(np.array(fig.data[0].y), y)

def test_line_chart_title_applied(viz):
    """The title kwarg must be forwarded (parallels heatmap's title test)."""
    fig = viz.line_chart(np.arange(3), {"s": np.arange(3)}, title="My Chart")
    assert fig.layout.title.text == "My Chart"

# ── image_grid cosmetic kwargs ───────────────────────────────────────────

def test_image_grid_suptitle_applied(viz):
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    fig = viz.image_grid(images, titles=["x"], suptitle="Overall Title")
    assert fig.layout.title.text == "Overall Title"

def test_image_grid_respects_height(viz):
    images = [np.zeros((4, 4, 3), dtype=np.uint8)]
    fig = viz.image_grid(images, titles=["x"], height=512)
    assert fig.layout.height == 512
