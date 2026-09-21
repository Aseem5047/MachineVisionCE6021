# Lab 00B — Gaussian Kernels, Convolution & Interpolation

## Introduction

This is the second introductory lab, building on Lab 00's class/method workflow. Instead of calling `cv2.GaussianBlur` as a black box, here you build the Gaussian filter kernel yourself from its 1D formula, apply it as a 2D convolution using `skimage`, and then use interpolation to resample an image. Putting the three together lets you see, directly, why images are blurred *before* they're shrunk: interpolating without a pre-filter first produces **aliasing** — spurious, wrong-looking patterns that weren't in the original image — while pre-filtering with your own Gaussian kernel removes it.

## Dataset / Test Images

No files from `Data/` are needed for this lab. The evaluation script instead generates a **chirp ("zone plate") test pattern** at runtime:

```python
x = np.arange(500) / 500 - 0.5
y = np.arange(500) / 500 - 0.5
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
f0, k = 10, 250
pattern = np.sin(np.pi * 2 * (f0 * R + k * R**2 / 2))
```

This is a 2D sinusoid whose local spatial frequency increases with distance from the centre — low frequency in the middle, sweeping up to just below the pattern's own Nyquist limit at the edges and corners. That continuous frequency sweep is exactly what makes it useful here: a real photograph is usually already band-limited by the camera optics and JPEG compression, so it often *won't* alias clearly or consistently when downsampled. A synthetic pattern with guaranteed high-frequency content gives a reliable, repeatable demonstration instead.

## Your Task

### Class: `ConvolutionInterpolator`

**Constructor:** `ConvolutionInterpolator()`

This class takes no constructor arguments — simply instantiate it and call the methods below.

### Methods to Implement

#### `gaussian_kernel_2d(sigma) → ndarray`

Build a 2D Gaussian kernel by taking the outer product of a 1D Gaussian with itself.

**Input:**
- `sigma` — standard deviation of the Gaussian.

**Output:** `K × K` kernel, normalised so it sums to 1 (so convolving with it doesn't change the image's overall brightness). There's no separate `K` input — the kernel size is *derived* from `sigma`, described below.

> **1D Gaussian:** g(x) = 1/(σ√(2π)) · exp(−x² / (2σ²)), sampled at the integer offsets x = −half, …, 0, …, half.
>
> **Rule of thumb for K:** a Gaussian's mass is concentrated within about ±3σ of its centre — beyond that, the tails contribute almost nothing. So rather than taking a kernel size as an input and hoping it's big enough, derive it directly from `sigma`: `half = ceil(3 * sigma)`, giving `K = 2 * half + 1`. This guarantees the sampled window always covers the full ±3σ span (rounding up, never truncating it), for any `sigma` you're given. Build the 2D kernel with `np.outer(g, g)`, then divide by its sum to normalise.

---

#### `convolve(image, kernel, **kwargs) → ndarray`

Convolve a 2D greyscale image with a kernel (e.g. the one from `gaussian_kernel_2d`).

**Input:**
- `image` — H × W greyscale array.
- `kernel` — Q × Q filter kernel.
- `mode` *(keyword, default `"same"`)* — output size: `"same"`, `"valid"`, or `"full"`.
- `boundary` *(keyword, default `"symm"`)* — boundary/padding style: `"symm"`, `"fill"`, or `"wrap"`.
- `fillvalue` *(keyword, default `0`)* — constant used when `boundary="fill"`.

**Output:** Filtered image — same shape as the input under the default `mode="same"`; a different shape for `"valid"`/`"full"`.

> Use `scipy.signal.convolve2d(image, kernel, mode=mode, boundary=boundary, fillvalue=fillvalue)`. This is *true* convolution (the kernel is flipped before sliding), unlike a correlation — though for a symmetric kernel like a Gaussian, flipping makes no numerical difference at all.
>
> **Output size (`mode`):** `"same"` (the default here) pads the input so the output matches its size — this is what you want for a filter you intend to compare against the original image. `"valid"` applies the kernel only where it fully overlaps the image, with no padding at all, so the output shrinks by `K − 1` pixels in each dimension. `"full"` is the reverse — every possible overlap, so the output grows.
>
> **Choice of padding (`boundary`, only relevant when `mode="same"` or `"full"`):** the border pixels need *something* to be filtered against beyond the image's edge. `"symm"` (the default here) mirrors nearby pixels across the border, which blends naturally into the image content. `"fill"` pads with a constant (`fillvalue`, default 0) instead, which drags the average down and darkens the edges. `"wrap"` treats the image as periodic, wrapping around to the opposite edge. For this lab, `"symm"` is the right default: it avoids the artificial dark border `"fill"` would introduce.

---

#### `interpolate(image, **kwargs) → ndarray`

Resample a 2D greyscale image by a scale factor.

**Input:**
- `image` — H × W greyscale array.
- `scale_factor` *(keyword, default `0.5`)* — output size relative to input; `< 1` downsamples, `> 1` upsamples.
- `order` *(keyword, default `0`)* — interpolation order (0 = nearest neighbour, 1 = bilinear, 3 = bicubic, …).

**Output:** Resampled image, shape scaled by `scale_factor`.

> Use `skimage.transform.rescale(image, scale_factor, order=order, anti_aliasing=False, mode='reflect')`.
>
> **`anti_aliasing=False` is not optional here.** By default, `rescale` silently applies its *own* Gaussian pre-filter whenever it detects downsampling — which would defeat the entire point of this lab, since you wouldn't be able to see the difference your own filter makes. This method must do interpolation *only*: no implicit smoothing of its own. That's also why the default `order=0` (nearest neighbour) is deliberate — it has no smoothing effect either, so whatever aliasing appears (or doesn't) in the output comes entirely from whether the input was filtered before being passed in.

## What the Pytests Check

Run the automated tests with:

```bash
pytest test_lab00B.py
```

The tests recompute each expected result independently and check values directly, rather than relying on visual inspection:

- **`gaussian_kernel_2d`** — returns a `K × K` float array whose size follows the `±3σ` rule of thumb; sums to 1; its values match a reference built from the 1D formula and `np.outer`; its peak is at the centre.
- **`convolve`** — matches `scipy.signal.convolve2d` called directly with the same arguments; preserves the input shape by default; shrinks correctly under `mode="valid"`; respects the `boundary` kwarg; leaves a flat (constant) region unchanged.
- **`interpolate`** — output shape matches `scale_factor`; values match `skimage.transform.rescale` called directly with `anti_aliasing=False`; critically, the output does **not** match what `rescale` would produce with `anti_aliasing=True` — i.e. your implementation must not be smoothing on its own.

## Evaluation Script

`evaluate_lab00B.py` demonstrates the aliasing effect this lab is built around:

- Generates the chirp test pattern described above.
- **`run_aliasing_demo(pattern)`** is the piece you need to complete: instantiate `ConvolutionInterpolator`, then produce two results from the same pattern —
  1. `aliased` — the pattern interpolated *directly* (no pre-filter).
  2. `filtered` — the pattern convolved with a Gaussian kernel from `gaussian_kernel_2d`, *then* interpolated by the same amount.
- Builds a 3-panel comparison figure (original pattern / aliased / filtered) using the provided `plot_aliasing_comparison()` helper — you don't need to write any Plotly code for this lab, just call it with your two results.
- Saves the figure to `lab00B_results.html`.

Compare the "Direct interpolation" and "Gaussian-filtered interpolation" panels once you've run it: the direct version shows visible aliasing — new, spurious ring/moiré patterns that aren't in the original chirp at all, especially near the edges and corners where the local frequency was highest — while the filtered version renders much closer to what a properly downsampled version of the pattern should look like.

Run it with:

```bash
python evaluate_lab00B.py
```

Then open `lab00B_results.html` in your browser — in VS Code / Codespaces, right-click the file → **Open with Live Server**, or run `python -m http.server 5000` and open the forwarded port.
