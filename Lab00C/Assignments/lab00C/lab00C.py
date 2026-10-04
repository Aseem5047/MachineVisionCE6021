# -*- coding: utf-8 -*-

"""Lab 00C — Introduction: Visualizing Results with Plotly.

This lab is a companion to Lab 00. Instead of an image-processing task, the
goal is to get comfortable with the Plotly charting patterns you will reuse
in every later lab: multi-panel image grids, heatmaps, and line charts, all
combined into a single shareable HTML report.

Task
----
Implement the three methods inside the ResultsVisualizer class:

  image_grid(images, titles=None, **kwargs)
      Arrange a list of images side by side in one figure using
      plotly.subplots.make_subplots and go.Image.

  heatmap(matrix, **kwargs)
      Display a 2D numeric array (e.g. a frequency-magnitude spectrum) as a
      go.Heatmap figure.

  line_chart(x, series, **kwargs)
      Plot one or more named 1D series against a shared x-axis using
      go.Scatter traces with a legend.

Each method must return a plotly.graph_objects.Figure — it should NOT call
fig.show() or write any files (that happens in evaluate_lab00C.py).
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResultsVisualizer:
    """Small helper class for building the Plotly figures used throughout
    this course: image grids, heatmaps, and line charts.

    No constructor arguments are needed — just instantiate and call the
    methods below.

    Example Usage
    ------------
    >>> viz = ResultsVisualizer()
    >>> fig = viz.image_grid([img1, img2], titles=["Before", "After"])
    """

    def image_grid(self, images, titles=None, **kwargs):
        """Arrange images side by side in a single figure.

        Args:
            images (list[ndarray]): H x W x 3 (or H x W) images to display.
            titles (list[str], optional): Subplot title per image. Defaults
                to "Image 1", "Image 2", ... if not given.
            **kwargs:
                suptitle (str): Overall figure title (default "").
                height   (int): Figure height in pixels (default 350).

        Returns:
            plotly.graph_objects.Figure: Figure with one go.Image panel per
            input image, arranged in a single row.
        """

        if titles is None:
            titles = [f"Image {i + 1}" for i in range(len(images))]

        suptitle = kwargs.get("suptitle", "")
        height = kwargs.get("height", 350)

        fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)

        for col, image in enumerate(images, start=1):
            fig.add_trace(go.Image(z=image), row=1, col=col)

        fig.update_layout(title_text=suptitle, height=height)

        return fig

    def heatmap(self, matrix, **kwargs):
        """Display a 2D numeric array as a heatmap.

        Args:
            matrix (ndarray): 2D array of numeric values.
            **kwargs:
                title      (str): Figure title (default "").
                colorscale (str): Plotly colourscale name (default "Viridis").

        Returns:
            plotly.graph_objects.Figure: Figure containing a single
            go.Heatmap trace.
        """

        title = kwargs.get("title", "")
        colorscale = kwargs.get("colorscale", "Viridis")

        fig = go.Figure(go.Heatmap(z=matrix, colorscale=colorscale))

        fig.update_layout(title_text=title)

        return fig

    def line_chart(self, x, series, **kwargs):
        """Plot one or more named series against a shared x-axis.

        Args:
            x (array-like): Shared 1D x-axis values.
            series (dict[str, array-like]): Maps trace name -> 1D y-values
                (same length as x).
            **kwargs:
                title       (str): Figure title (default "").
                xaxis_title (str): X-axis label (default "x").
                yaxis_title (str): Y-axis label (default "y").

        Returns:
            plotly.graph_objects.Figure: Figure with one go.Scatter line
            trace per entry in `series`, with a legend.
        """

        title = kwargs.get("title", "")
        xaxis_title = kwargs.get("xaxis_title", "x")
        yaxis_title = kwargs.get("yaxis_title", "y")

        fig = go.Figure()

        for name, y in series.items():
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))

        fig.update_layout(title_text=title,
                          xaxis_title=xaxis_title,
                          yaxis_title=yaxis_title)

        return fig