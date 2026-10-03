# Lab 00C — Visualizing Results with Plotly

## Introduction

This is a companion to Lab 00, and the third of the introductory labs. Instead of an image-processing algorithm, the task here is to get comfortable with the Plotly charting patterns you will reuse in **every** later lab: side-by-side image grids, heatmaps, and line charts, all combined into a single shareable HTML report. Later labs strip out the plotting code and leave you a bare `# TODO: Add your own visualisation` — this lab is where you actually learn the API you'll need to fill that in.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| `Great_Sphinx_of_Giza_-_20080716a.jpg` | Same test image as Lab 00. Located in `Data/`. |
| `960px-Le_sacre_Coeur_bordercropped.jpg` | Same test image as Lab 00. Located in `Data/`. |

## Your Task

### Class: `ResultsVisualizer`

**Constructor:** `ResultsVisualizer()`

This class takes no constructor arguments — simply instantiate it and call the methods below. In this lab, instantiating the class is already done for you in `evaluate_lab00C.py`; you only need to implement the methods themselves.

### Methods to Implement

#### `image_grid(images, titles=None, **kwargs) → go.Figure`

Arrange a list of images side by side in a single figure.

**Input:**
- `images` — list of H × W × 3 (or H × W) images.
- `titles` *(optional)* — list of subplot title strings, one per image. Defaults to `"Image 1"`, `"Image 2"`, … if omitted.
- `suptitle` *(keyword, default `""`)* — overall figure title.
- `height` *(keyword, default `350`)* — figure height in pixels.

**Output:** A `plotly.graph_objects.Figure` with one `go.Image` panel per input image.

> Use `plotly.subplots.make_subplots` to lay out the panels, and `go.Image(z=image)` to add each one as a trace.

---

#### `heatmap(matrix, **kwargs) → go.Figure`

Display a 2D numeric array as a heatmap — the same chart type later labs use for response maps and similarity matrices.

**Input:**
- `matrix` — 2D numpy array of numeric values.
- `title` *(keyword, default `""`)* — figure title.
- `colorscale` *(keyword, default `"Viridis"`)* — Plotly colourscale name.

**Output:** A `plotly.graph_objects.Figure` containing a single `go.Heatmap` trace.

---

#### `line_chart(x, series, **kwargs) → go.Figure`

Plot one or more named series against a shared x-axis — the pattern later labs use for ROC curves and threshold sweeps.

**Input:**
- `x` — shared 1D array of x-axis values.
- `series` — `dict` mapping trace name → 1D y-values (same length as `x`).
- `title` *(keyword, default `""`)* — figure title.
- `xaxis_title` *(keyword, default `"x"`)* — x-axis label.
- `yaxis_title` *(keyword, default `"y"`)* — y-axis label.

**Output:** A `plotly.graph_objects.Figure` with one `go.Scatter` line trace per entry in `series`, with a legend.

> None of these methods should call `fig.show()` or write files — just build and return the `Figure`. Saving to HTML happens in `evaluate_lab00C.py`.

## Background: The Plotly Building Blocks

Every method you implement follows the same pattern: build one or more **traces**, add them to a **figure**, then set figure-level properties with `update_layout`. This section walks through the pieces you'll use.

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

- **`plotly.graph_objects`** (imported as `go`) is Plotly's object-oriented API — each chart type (image, heatmap, line, …) has a corresponding `go.*` class that you construct with the data you want plotted.
- **`plotly.subplots.make_subplots`** builds a `Figure` that already has a grid of panels laid out inside it, so you can add different traces to different panels of the same figure.

**What is a trace?**

A **trace** is a single layer of plotted data — one image panel, one heatmap, one line. A `Figure` is a container that can hold one or more traces (plus layout settings). You add a trace to a figure with `fig.add_trace(...)`, or, for a single-trace figure, you can pass it directly to `go.Figure(...)`:

```python
fig = go.Figure(go.Heatmap(z=matrix))       # one trace, built directly
fig.add_trace(go.Scatter(x=x, y=y))         # ...or added onto an existing figure
```

**The trace types used in this lab**

| Trace | Used in | What it plots |
|-------|---------|----------------|
| `go.Image(z=image)` | `image_grid` | Renders an H×W×3 (or H×W) array as an actual image panel — `z` is the pixel array itself, not numeric data to be colour-mapped. |
| `go.Heatmap(z=matrix, colorscale=...)` | `heatmap` | Colour-maps a 2D numeric array — each value in `z` is mapped to a colour via `colorscale` (e.g. `"Viridis"`). Unlike `go.Image`, this is for arbitrary numeric data, not pixel values. |
| `go.Scatter(x=x, y=y, mode='lines', name=...)` | `line_chart` | Plots `y` against `x`. `mode='lines'` draws it as a line chart (rather than markers); `name` sets the label shown in the legend. |

**Laying out multiple panels with `make_subplots`**

`image_grid` needs one panel per input image side by side, which is exactly what `make_subplots` is for:

```python
fig = make_subplots(rows=1, cols=len(images), subplot_titles=titles)
for col, image in enumerate(images, start=1):
    fig.add_trace(go.Image(z=image), row=1, col=col)
```

`rows`/`cols` define the grid, `subplot_titles` labels each panel, and the `row=`/`col=` arguments to `add_trace` say which panel a given trace belongs to. `heatmap` and `line_chart` only ever need a single panel, so they build a plain `go.Figure()` instead.

**`update_layout` — figure-level settings**

Once the traces are in place, `fig.update_layout(...)` sets properties that apply to the whole figure rather than to any one trace: the overall title (`title_text`), axis labels (`xaxis_title`, `yaxis_title`), figure size (`height`), margins, and so on:

```python
fig.update_layout(title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
```

It's called *after* the traces are added, and can be called as many times as convenient — each call only updates the properties you pass in, leaving everything else unchanged.

## Step-by-Step Walkthrough

This section works through each method in full detail.

### `image_grid(images, titles=None, **kwargs)`

**Goal:** one figure, with the images laid out side by side in a single row.

1. **Build a default list of titles for the case where `titles` is not input.** If `titles` wasn't supplied (it's `None`), build a default list yourself: `"Image 1"` for the first image, `"Image 2"` for the second, and so on. A list comprehension over `enumerate(images)` is a natural fit:
   ```python
   if titles is None:
       titles = [f"Image {i + 1}" for i in range(len(images))]
   ```
2. **Setup Kwargs.** `suptitle` and `height` live in `kwargs`, not as named parameters — read them with `kwargs.get('suptitle', "")` and `kwargs.get('height', 350)`.
3. **Create the grid.** Call `make_subplots(rows=1, cols=len(images), subplot_titles=titles)` — this gives you an empty figure with one panel per image, each panel already labelled from `titles`.
4. **Fill in each panel.** Loop over `images`, keeping track of which column you're on (`enumerate(images, start=1)` is convenient, since Plotly's columns are 1-indexed). For each image, call `fig.add_trace(go.Image(z=image), row=1, col=col)`:
   ```python
   for col, image in enumerate(images, start=1):
       fig.add_trace(go.Image(z=image), row=1, col=col)
   ```
5. **Set the overall title and size.** `fig.update_layout(title_text=suptitle, height=height)`.
6. **Return `fig`.**

### `heatmap(matrix, **kwargs)`

**Goal:** one figure, with a single heatmap panel.

1. **Setup Kwargs.** `title` (default `""`) and `colorscale` (default `"Viridis"`), both via `kwargs.get(...)`.
2. **Build the figure and its one trace together.** Since there's only ever one panel, you don't need `make_subplots` here — construct the trace and hand it straight to `go.Figure(...)`: `go.Figure(go.Heatmap(z=matrix, colorscale=colorscale))`.
3. **Set the title.** `fig.update_layout(title_text=title)`.
4. **Return `fig`.**

### `line_chart(x, series, **kwargs)`

**Goal:** one figure, with one line per entry in `series`, all sharing the same x-axis.

1. **Setup Kwargs.** `title`, `xaxis_title` (default `"x"`), `yaxis_title` (default `"y"`).
2. **Start with an empty figure.** `fig = go.Figure()` — there's no trace to pass in yet, since the number of lines depends on how many entries are in `series`.
3. **Add one trace per series.** `series` is a `dict` mapping `name -> y_values`. Loop over `series.items()`, and for each `(name, y)` pair call `fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name))`. Every trace shares the same `x` — only `y` and `name` change per loop iteration.
4. **Set the title and axis labels.** `fig.update_layout(title_text=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)`.
5. **Return `fig`.**

---

Across all three methods, the shape is identical: **unpack the kwargs you need → build the trace(s) → assemble/label the figure → return it.** Once one method is working, the other two are the same pattern with a different trace type.

## What the Pytests Check

Run the automated tests with:

```bash
pytest test_lab00C.py
```

The tests check the *structure* of each returned figure:

- **`image_grid`** — returns a `Figure`; contains exactly one `go.Image` trace per input image; uses default `"Image N"` titles when none are given.
- **`heatmap`** — returns a `Figure` with exactly one `go.Heatmap` trace; the trace's `z` values match the input matrix; the title is applied.
- **`line_chart`** — returns one `go.Scatter` trace per entry in `series`; trace names match the `series` keys; trace `x`/`y` values match the input; axis titles are applied.

## Evaluation Script

The `evaluate_lab00C.py` will demonstrate your implementation by:

- Loading the Sphinx and Sacré-Cœur images from `Data/` (same images as Lab 00).
- Building an **image grid** comparing the two test images.
- Computing the FFT log-magnitude spectrum of the Sphinx image and displaying it as a **heatmap**.
- Extracting the horizontal intensity profile through the middle row of each image and plotting them as a **line chart**.
- Saving all three figures to a single `lab00C_results.html` file.

### Helper Functions in `evaluate_lab00C.py`

A few small helpers do the data prep so `evaluate()` can focus on calling your `ResultsVisualizer` methods:

- **`_resize_for_display(image, max_width=640)`** — downscales an image to a maximum width (keeping aspect ratio) so browser rendering doesn't alias the plot.
- **`_fft_magnitude(image)`** — converts to greyscale, applies a 2D FFT, and returns the shifted log-magnitude spectrum — the array that gets passed to `heatmap()`.
- **`_row_profile(image, row_frac=0.5)`** — converts to greyscale and returns the pixel intensities along one horizontal row (the middle row by default) — the array that gets passed to `line_chart()`.

### Saving the Report with `save_html`

Writing the HTML file itself isn't done locally — `evaluate_lab00C.py` imports the same helper every lab in this course uses:

```python
from helpers.plotting import save_html
...
save_html(*figs, output_path=os.path.join(LAB_DIR, "lab00C_results.html"))
```

`save_html(*figs, output_path)` combines any number of Plotly figures into a single self-contained HTML page. It calls `plotly.io.to_html(fig, full_html=False, ...)` on each figure to get an embeddable `<div>` snippet (rather than a full standalone page per figure), then stitches those snippets together inside one minimal HTML shell, separated by `<hr>` rules. Only the *first* figure is given `include_plotlyjs="cdn"` — that pulls in the Plotly JavaScript library once from a CDN — while the rest set it to `False` so the library isn't downloaded and embedded redundantly for every chart. The result is written to `output_path`, and the function prints that path plus two ways to open it (VS Code Live Server, or Python's built-in HTTP server) — useful in Codespaces, where you can't just double-click a local file.

Run it with:

```bash
python evaluate_lab00C.py
```

Then open `lab00C_results.html` in your browser — in VS Code / Codespaces, right-click the file → **Open with Live Server**, or run `python -m http.server 5000` and open the forwarded port.
