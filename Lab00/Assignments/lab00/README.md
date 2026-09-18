# Lab 00 — Simple Image Processing

## Introduction

This introductory lab gets you comfortable with the lab workflow before tackling more advanced topics.
You will implement two fundamental spatial-domain image filters — Gaussian blur and unsharp-mask sharpening — using OpenCV.
The goal is to understand how convolution-based filters affect the frequency content of an image and to practise running the evaluation script.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| `Great_Sphinx_of_Giza_-_20080716a.jpg` | High-resolution photograph of the Great Sphinx; used as the primary test image. Located in `Data/`. |
| `960px-Le_sacre_Coeur_bordercropped.jpg` | Photograph of Sacré-Cœur basilica; used as a second test image. Located in `Data/`. |

## Your Task

### Class: `SimpleImageProcessing`

**Constructor:** `SimpleImageProcessing()`

This class takes no constructor arguments — simply instantiate it and call the methods below. In this lab, instantiating the class is already done for you in `evaluate_lab00.py`; you only need to implement the methods themselves.

### Methods to Implement

#### `add_blur(image, **kwargs) → ndarray`

Apply a Gaussian blur to the input image.

**Input:**
- `image` — H × W × 3 (or H × W) uint8 numpy array.
- `ksize` *(keyword, default 15)* — Gaussian kernel size; must be a positive odd integer.

**Output:** Blurred image as a uint8 numpy array with the same shape as the input.

> Use `cv2.GaussianBlur` with the kernel size drawn from `kwargs`.

---

#### `add_sharpen(image, **kwargs) → ndarray`

Sharpen the image using an **unsharp mask**: subtract a blurred copy of the image from the original, scaled by a strength factor.

**Input:**
- `image` — H × W × 3 (or H × W) uint8 numpy array.
- `ksize` *(keyword, default 15)* — Kernel size for the internal blur.
- `strength` *(keyword, default 1.5)* — How strongly to apply the sharpening effect.

**Output:** Sharpened image as a uint8 numpy array with the same shape as the input.

> `cv2.addWeighted` can combine the original and blurred images in one step. Remember to clip the result to [0, 255] before casting to uint8.

## What the Pytests Tests Check

Run the automated tests with:

```bash
pytest test_lab00.py
```

The tests will verify:

- **Blur reduces high-frequency energy** — After blurring, the total energy in the high-frequency half of the FFT magnitude spectrum is lower than in the original image. Tested on both the Sphinx and Sacré-Cœur images.
- **Sharpen increases the HF-to-LF energy ratio** — After sharpening, the ratio of high-frequency energy to low-frequency energy is greater than that of the original image. Tested on both images.
- **Output shape matches input** — The returned array has the same height, width, and channel count as the input.
- **Output dtype is uint8** — The returned array must be `np.uint8`.

## Evaluation Script

The `evaluate_lab00.py` will demonstrate your implementation by:

- Loading both the Sphinx and Sacré-Cœur images from `Data/` using the `load_image` helper.
- Instantiating `SimpleImageProcessing`.
- Calling `add_blur` and `add_sharpen` on each image with default parameters.
- Producing a three-panel figure for each image showing: **Original | Blurred | Sharpened**.
- Saving both figures to a single `lab00_results.html` file.

Save all Plotly figures to a single HTML file using the provided `_save_html()` helper.

Run it with:

```bash
python evaluate_lab00.py
```

## Background: Classes in Python

A `class` is a blueprint for an object that bundles together data (attributes) and behaviour (methods). In this lab you work with `SimpleImageProcessing`:

```python
class SimpleImageProcessing:
    def add_blur(self, image, **kwargs):
        ...
```

- `self` refers to the specific instance the method is called on. It's the first parameter of every instance method, but you never pass it explicitly — Python fills it in for you.
- You create (**instantiate**) an object from the class, then call methods on it with dot notation:
  ```python
  proc = SimpleImageProcessing()
  blurred = proc.add_blur(image, ksize=21)
  ```
  Here `proc` is the instance; `proc.add_blur(...)` is equivalent to `SimpleImageProcessing.add_blur(proc, image, ksize=21)` — Python passes `proc` in as `self` automatically. In this lab, this step is already performed for you in `evaluate_lab00.py` — it's shown here so you understand what that code is doing.

**Positional vs. keyword arguments**

- A **positional argument** is matched to a parameter by its position in the call, e.g. `image` above — it must always be supplied, and in the right place.
- A **keyword argument** is matched by name, e.g. `ksize=21`. Keyword arguments can be given in any order and are often optional, since the function can supply a default when one isn't passed.

**`**kwargs`**

`add_blur(self, image, **kwargs)` collects *any* keyword arguments passed at the call site into a dictionary named `kwargs`. Inside the method, `kwargs.get('ksize', 15)` looks up `'ksize'` in that dictionary and falls back to `15` if it wasn't supplied. This gives the method flexibility:

```python
proc.add_blur(image)             # kwargs = {}              -> ksize defaults to 15
proc.add_blur(image, ksize=21)   # kwargs = {'ksize': 21}    -> ksize = 21
```

The method signature doesn't need to change to support new optional parameters — callers only need to specify the keywords they care about, which is why both `add_blur` and `add_sharpen` in this lab are written this way.

## Background: NumPy Arrays & OpenCV

A **NumPy array** (`numpy.ndarray`) is a grid of numbers, all of the same fixed data type, arranged along one or more axes. A **2D array** has two axes — think of it as a table of rows and columns, addressed as `array[row, col]`. A greyscale image is naturally a 2D array of shape `(H, W)`: one number per pixel, its brightness. A colour image adds a third axis for channels, giving a 3D array of shape `(H, W, 3)` — the method docstrings in this lab write that as "H × W × 3 (or H × W)" to cover both cases.

This matters here because **OpenCV's Python functions consume and return plain NumPy arrays** — there's no separate "image object" to convert to or from. `image` in `add_blur`/`add_sharpen` is just an ndarray (produced by `load_image`), and `cv2.GaussianBlur(image, ...)` / `cv2.addWeighted(...)` both take ndarrays in and hand ndarrays back out. That's also why NumPy functions (`np.clip`, array `.astype(...)`, etc.) can be applied directly to OpenCV's output with no extra conversion step.

Two NumPy operations used in `add_sharpen` are worth knowing:

- **`np.clip(array, min, max)`** — caps every value in the array to lie within `[min, max]`, leaving values already inside that range unchanged. `add_sharpen`'s weighted sum can push pixel values below 0 or above 255, so the result is clipped to `[0, 255]` before it's treated as a valid image again.
- **`.astype(np.uint8)`** — casts an array to a new dtype, here 8-bit unsigned integer (the standard dtype for pixel values, range 0–255). This cast **must** happen after clipping, not before: `uint8` doesn't clamp out-of-range values, it wraps them (e.g. `256` becomes `0`, `-1` becomes `255`), which would silently corrupt the image instead of just capping brightness.

## Background: The Unsharp Mask

**Unsharp masking** is a classic sharpening technique — the name comes from a darkroom trick where a blurred ("unsharp") copy of a photo negative was used to boost the sharp original. In digital image processing it works the same way, computationally:

1. Make a blurred copy of the image using a Gaussian blur. Because a Gaussian blur is a low-pass filter, this blurred copy contains mostly the image's **low-frequency** content (smooth regions, gradual shading) and has lost most of its **high-frequency** content (edges, fine detail).
2. Subtracting the blurred copy from the original therefore leaves behind an estimate of just the high-frequency detail.
3. Adding that detail back on top of the original, scaled by a `strength` factor, exaggerates edges and fine texture — the image looks sharper.

As a formula, with `strength` controlling the effect:

```
sharpened = original + strength × (original − blurred)
          = (1 + strength) × original − strength × blurred
```

This is exactly what `cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)` computes in one call — it forms a weighted sum of `image` and `blurred` with the weights `(1 + strength)` and `-strength` respectively.

A couple of practical points that show up in `add_sharpen`:

- `ksize` controls how strongly the internal copy is blurred, which in turn controls what counts as "high frequency" detail to be boosted — a larger `ksize` blurs more aggressively, so the mask picks up coarser detail.
- Because amplifying high-frequency content can push pixel values below 0 or above 255, the result must be clipped back into the valid `[0, 255]` range before casting to `uint8`.

