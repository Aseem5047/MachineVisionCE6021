# Lab 01 — Efficient Separable Gaussian Convolution

## Introduction

Gaussian convolution is one of the most common operations in image processing, underpinning blur, edge detection, and scale-space methods.
This lab challenges you to implement it as efficiently as possible by exploiting the separability of the 2D Gaussian kernel: a single 2D convolution is replaced by two 1D passes.
Your implementation is benchmarked against time thresholds, so performance matters as much as correctness.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| `2560px-Great_Sphinx_of_Giza_-_20080716a.jpg` | High-resolution Sphinx photograph, loaded at half scale and converted to greyscale for testing. Located in `Data/`. |

## lab01.py Starter Code

### Class: `GaussianFilt`

**Constructor:** `GaussianFilt(sigma)`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sigma` | float | — | Standard deviation of the Gaussian kernel. Controls smoothing strength. |

### Methods to Implement

#### `gauss_kernel() → ndarray`

Generate a normalised 1D Gaussian kernel of appropriate size for `self.sigma`.

**Output:** Shape `(k_size, 1)` column-vector kernel (float64), where `k_size = int(6 * sigma + 1)`.

> Compute `k_size` from `sigma`, then evaluate the 1D Gaussian PDF at each integer offset. Don't forget to expand the result to a column vector with `np.expand_dims`.

---

#### `my_conv_method(image) → ndarray`

Perform a full separable 2D Gaussian convolution using `gauss_kernel()`.

**Input:** `image` — 2D greyscale image (H, W), any numeric dtype.

**Output:** Blurred image of the same shape, dtype **uint8**.

> The most straightforward approach to performing separable convolution is to convolve the image with the 1D kernel along one axis, then convolve the result with the same kernel (transposed) along the other axis.
>
> To keep the output the same size as the input, pad the image once — by `k_size // 2` on every side — before either pass, then perform both 1D convolutions as 'valid' (no need to re-pad between passes).

## What the Pytest Checks

Run the automated tests with:

```bash
pytest test_lab01.py
```

The tests will verify:

- **Execution time — relative to a naive baseline:** the tests time a direct 2D `convolve2d` using the full (outer-product) Gaussian kernel — the "obviously correct but not optimised" approach this lab asks you to beat — and check your `my_conv_method` runs within a fraction of that baseline time, at five increasingly strict tiers: **1.0×**, **0.75×**, **0.5×**, **0.25×**, and **0.1×** the baseline. Because the threshold is relative rather than a fixed number of seconds, it adapts automatically to however fast or slow the machine running the tests is.
- **Gaussian kernel shape** — `gauss_kernel()` must return a 1D kernel (as a column vector) of length `int(6 * sigma + 1)`.
- **Output shape** — returned array must have the same (H, W) dimensions as the input greyscale image.
- **Output dtype** — returned array must be `np.uint8`.

## Evaluation Script

The supplied `evaluate_lab01.py` will demonstrate your implementation by:

- Loading the Sphinx image at half scale and converting to greyscale.
- Running `my_conv_method` for one or more sigma values (e.g. σ = 2, 5, 10).
- Producing a multi-panel figure showing the **original image** alongside each **blurred result**.
- Printing the execution time for each convolution to the terminal.
- Saving all figures to `lab01_results.html`.

Run it with:

```bash
python evaluate_lab01.py
```
