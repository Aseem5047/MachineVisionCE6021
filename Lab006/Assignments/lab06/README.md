# Lab 06 — Scale Invariant Blob Detector

## Introduction

<p align="center">
  <img src="https://github.com/tonyscan6003/etivities/blob/main/sunflowers_temp.jpg?raw=true" alt="A field of sunflowers, small in the distance and large in the foreground" width="420">
  <br><em>The sunflowers used in this lab.</em>
</p>

How large is each flower head in this picture? The sunflowers in the distance are only a few pixels across, while those in the foreground are a hundred pixels or more. A detector that looks for features of one fixed size would find some of them and miss the rest.

The **Scale Invariant Blob Detector** solves this by searching not only across the image (row, col) but also across *scale*. The image is filtered with a **Laplacian of Gaussian** (LoG) at a whole range of Gaussian widths, and the responses are stacked into a 3D **scale space**. A blob appears as a local maximum in this `(row, col, scale)` volume, and the scale at which it peaks gives its size.

The lab is split into ten small steps. Each step is one short method, and each one can be checked before the next is started.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| Sunflowers image | Downloaded automatically from `FLOWERS_IMAGE_URL` and cached in `Data/`. Converted to greyscale before use. 768 × 1024 pixels, with dark flower centres of many different sizes. |

## How This Lab Works

1. **One step** is implemented in `lab06.py`.
2. `python evaluate_lab06.py` is run. Every finished step is run, a figure for each one is added to `lab06_results.html`, and the script stops at the first step that has not yet been implemented.
3. `lab06_results.html` is opened and the new figure is checked before the next step is started.

`pytest test_lab06.py` can also be run at any point — the tests are grouped by step, so the step that needs attention is indicated by the failing tests. The tests use small synthetic images, so the whole suite finishes in a few seconds.

> **Runtime:** from Step 5 onwards the full scale space (33 scales) is built on the sunflowers photo, which takes around 20–30 seconds; a `tqdm` progress bar shows that it is still working. It is built twice (Steps 5 and 10), so a complete run of the evaluate script takes about a minute.

> **Image coordinates:** NumPy arrays are indexed `image[row, col]`, with row 0 at the **top**, and the scale space is indexed `scale_space[row, col, k]`. Points are passed to OpenCV drawing functions such as `cv2.circle` as `(x, y)` = `(col, row)` — the other way round.

## The Pipeline at a Glance

```
greyscale image
  → 1. gauss_kernel             1D Gaussian kernel
  → 2. gauss_2nd_deriv_kernel   1D second-derivative-of-Gaussian kernel
  → 3. second_derivatives       Lxx, Lyy by separable convolution
  → 4. laplacian                sigma² · (Lxx + Lyy) at one scale
  → 5. build_scale_space        Step 4 at every scale -> (rows, cols, scales) volume
  → 6. candidate_points         points above the threshold
  → 7. lap_max_det              is one point larger than its 26 neighbours?
  → 8. find_maxima              Step 7 at every candidate -> [magnitude, row, col, sigma]
  → 9. blob_circles             centre and radius √2 · sigma of each blob
  → 10. detect_blobs            runs steps 5, 8 and 9 in order
```

## Background

### The Laplacian of Gaussian (LoG)
The **Laplacian** `Lxx + Lyy` — the sum of the second derivatives in x and y — responds strongly wherever the intensity curves sharply, as it does at the centre of a blob. The sign of the response depends on whether the blob is darker or brighter than its surroundings; the dark sunflower centres give a positive response. The Laplacian is applied to a **Gaussian-smoothed** image rather than the raw image. This suppresses noise and ties the filter to a **scale**: after smoothing with a wide Gaussian only large blobs remain, and after a narrow one small blobs still respond.

### Separable kernels
Smoothing with a 2D Gaussian and then taking the second derivative in x is the same as convolving with the second derivative of a 1D Gaussian along x and a plain 1D Gaussian along y. Two 1D convolutions are much cheaper than one 2D convolution of the same size — the same idea as in Lab 01. The two 1D kernels are built in Steps 1–2 and combined in Step 3.

### Why multiply by sigma²?
The raw LoG response becomes *weaker* as sigma grows, because the same change in intensity is spread over a larger area. Multiplying by `sigma²` (the **scale-normalised** Laplacian, Step 4) removes this effect, so responses at different scales can be compared directly. Without it, the detector would favour small scales.

### Scale space and 26-neighbour local maxima
Stacking the LoG response over many scales (Step 5) gives a 3D volume indexed by `(row, col, k)`, where `k` is the index of the scale. A blob gives its strongest response at its own scale, so a blob is a point that is larger than all **26 neighbours** in the surrounding 3 × 3 × 3 block — the 8 neighbours in its own layer and the 9 in each of the layers above and below (Step 7).

### Candidates and threshold
Points with a weak response cannot be blobs, so the points above the threshold `thresh` are picked out first (Step 6) and only those are tested. Points on the outer layer of the volume (first or last row, column or scale) are left out, since they do not have a full 3 × 3 × 3 neighbourhood.

### From sigma to radius
A disc of radius `r` gives its strongest scale-normalised LoG response at `sigma = r / √2`. Each detected maximum is therefore drawn as a circle of radius `√2 · sigma` (Step 9).

## Your Task

### Class: `ScaleInvDet`

**Constructor:** `ScaleInvDet(**kwargs)` — all settings have defaults, so `ScaleInvDet()` works with no arguments. The class is instantiated in `evaluate_lab06.py`.

The constructor is **already written**. Every setting is stored on `self`, together with the list of scales worked out from them, and these attributes (e.g. `self.sigma_vals`, `self.thresh`) should be used in the methods rather than numbers typed in directly — this is checked by the tests, which change the settings.

| Setting | Default | Used in | Meaning |
|---------|---------|---------|---------|
| `self.sigma_min` | 1 | Step 5 | Smallest sigma in the scale space. |
| `self.sigma_max` | 36 | Step 5 | Largest sigma in the scale space. |
| `self.scale_fac` | 1.05 | Step 5 | Approximate spacing between consecutive sigma values. |
| `self.thresh` | 40 | Step 6 | Minimum response for a point to be a candidate. |
| `self.num_scales` | 33 *(derived)* | Step 5 | Number of scales, `int((sigma_max − sigma_min) / scale_fac)`. |
| `self.sigma_vals` | *(derived)* | Steps 5, 7 | The `num_scales` sigma values, evenly spaced from `sigma_min` to `sigma_max`. |

### Methods to Implement

| Step | Method | Input → Output |
|------|--------|----------------|
| 1 | `gauss_kernel(sigma)` | sigma → `(1, k)` Gaussian kernel |
| 2 | `gauss_2nd_deriv_kernel(sigma)` | sigma → `(1, k)` second-derivative kernel |
| 3 | `second_derivatives(image, sigma)` | image + sigma → `(Lxx, Lyy)` |
| 4 | `laplacian(image, sigma)` | image + sigma → LoG response |
| 5 | `build_scale_space(image)` | image → `(m, n, num_scales)` scale space |
| 6 | `candidate_points(scale_space)` | scale space → `(P, 3)` array of `(row, col, k)` |
| 7 | `lap_max_det(scale_space, row, col, k)` | scale space + one point → `[magnitude, row, col, sigma]` or `[]` |
| 8 | `find_maxima(scale_space)` | scale space → `(N, 4)` array of `[magnitude, row, col, sigma]` |
| 9 | `blob_circles(maxima)` | `(N, 4)` maxima → `(N, 3)` array of `[row, col, radius]` |
| 10 | `detect_blobs(image)` | image → `(N, 3)` array of `[row, col, radius]` |

## Step-by-Step Walkthrough

Each step below gives the goal, the details that the result depends on, and a hint — a function to look up, or an earlier lab where the same thing was done. The code itself is to be written from these.

### Step 1 — `gauss_kernel(sigma)`

**Goal:** a 1D Gaussian kernel, as a row vector of shape `(1, k)`.

- **Size:** `k = int(6σ + 1)`, with `half = (k − 1) // 2`. The positions `x` run from `−half` to `+half`.
- **Gaussian:** `G(x) = 1 / (σ√(2π)) · exp(−x² / (2σ²))`

**Hint:** this is the first part of Step 1 in Lab 05 (`gaussian_kernel`), stopping before the outer product. The `sigma` passed in is used here, not a setting on `self`. `np.expand_dims` adds the extra axis.

**Check:** a single smooth bump is shown in the Step 1 figure — 25 values for `sigma = 4`.

### Step 2 — `gauss_2nd_deriv_kernel(sigma)`

**Goal:** the second derivative of the 1D Gaussian, as a row vector of the same size.

The second derivative of the Gaussian is `G(x) · (x² − σ²) / σ⁴`, where `G(x)` is the Gaussian of Step 1.

**Hint:** the second derivative is calculated over the same range of `x` values as Step 1 (`−half` to `+half`, with `k = int(6σ + 1)`), and is returned as a row vector of shape `(1, k)` in the same way.

**Check:** in the Step 2 figure the new kernel is negative in the middle and crosses zero at `x = ±sigma`.

### Step 3 — `second_derivatives(image, sigma)`

**Goal:** `Lxx` and `Lyy`, each built from two 1D convolutions.

The two kernels come from Steps 1 and 2. A row vector filters along the rows, and its transpose filters down the columns:

| Output | First convolution | Second convolution |
|--------|-------------------|--------------------|
| `Lxx` | image with the second-derivative kernel, along the rows | result with the Gaussian, down the columns |
| `Lyy` | image with the second-derivative kernel, down the columns | result with the Gaussian, along the rows |

**Hint:** `signal.convolve2d` is used with `boundary='symm'` and `mode='same'`, as in Step 2 of Lab 05. `.T` gives the transpose of a kernel. Separable convolution was also the subject of Lab 01.

**Check:** vertical structure is picked out in `Lxx` and horizontal structure in `Lyy`.

### Step 4 — `laplacian(image, sigma)`

**Goal:** the scale-normalised LoG at one scale.

The response is `σ² · (Lxx + Lyy)`.

**Hint:** `Lxx` and `Lyy` come from the method written in Step 3.

**Check:** the response in the Step 4 figure (squared, so that the sign does not matter) is strongest on the flower centres of about the chosen size.

### Step 5 — `build_scale_space(image)`

**Goal:** the Step 4 response at every scale, stacked into one 3D array.

Layer `k` of the result is the Step 4 response at `self.sigma_vals[k]`, giving an `(m, n, num_scales)` array.

Every later step depends on this array, so the code is given in full:

```python
layers = []
for sigma in tqdm(self.sigma_vals, desc="Building scale space"):
    layers.append(self.laplacian(image, sigma))
return np.dstack(layers)
```

The loop calls Step 4 once per value in `self.sigma_vals` and collects the layers in a list; `np.dstack` then stacks them along a third axis. Wrapping the loop's sequence in `tqdm(...)` adds a progress bar, which is useful because this step is slow.

**Check:** in the Step 5 grid, small distant flowers respond at small sigma and the large foreground flowers at large sigma.

### Step 6 — `candidate_points(scale_space)`

**Goal:** the `(row, col, k)` of every interior point above `self.thresh`.

The outer layer of the volume (first and last row, column and scale) is left out, and the indices returned must refer to the **full** scale space.

**Hint:** a slice such as `[1:-1]` removes the first and last entries along an axis. `np.argwhere` returns one row of indices per `True` element of a boolean array. Those indices count from the start of the trimmed array, so 1 needs to be added to each of them, so that they point to the correct position in the full scale space used in later steps.

**Check:** the number of candidates at each scale is shown in the Step 6 bar chart.

### Step 7 — `lap_max_det(scale_space, row, col, k)`

**Goal:** whether one point is larger than all of its 26 neighbours.

The point's value is compared with the largest value in the 3 × 3 × 3 block centred on it (27 values, including the point itself). The point is a maximum if `value + 0.001` is greater than that largest value; the small `0.001` is added because the point is never *strictly* greater than itself.

- **If it is a maximum:** the list `[value, row, col, sigma]` is returned, where `sigma` is `self.sigma_vals[k]`.
- **Otherwise:** an empty list `[]` is returned.

**Hint:** a block can be cut out by slicing all three axes at once. A slice `a:b` stops *before* `b`, so three values centred on `row` end at `row + 2`. `np.max` gives the largest value in the block.

**Check:** in the Step 7 figure a small synthetic example is tested — the single peak at `(2, 2, 1)` is reported as a maximum, and the weaker point next to it is not.

### Step 8 — `find_maxima(scale_space)`

**Goal:** every candidate that is a local maximum.

Step 7 (`lap_max_det`) is applied to each candidate from Step 6 (`candidate_points`), and the results that are not empty are collected into an `(N, 4)` array. An empty `(0, 4)` array is returned when nothing is found, so that the output always has the same shape.

This step loops over a large number of candidates, so the code is given in full:

```python
maxima = []
for row, col, k in tqdm(self.candidate_points(scale_space), desc="Finding maxima"):
    result = self.lap_max_det(scale_space, row, col, k)
    if result:
        maxima.append(result)

if not maxima:
    return np.empty((0, 4))
return np.array(maxima)
```

`self.candidate_points(scale_space)` gives one `(row, col, k)` per candidate, and `self.lap_max_det` returns an empty list for a point that is not a maximum — an empty list counts as false in an `if` statement. Wrapping the candidates in `tqdm(...)` adds a progress bar. `np.empty((0, 4))` gives the empty case — `find_lines` in Lab 03 handled its empty case in the same way.

**Check:** in the Step 8 figure most of the red dots lie on flower centres.

### Step 9 — `blob_circles(maxima)`

**Goal:** a centre and radius for each blob.

The radius is `√2 · sigma` (column 3 of `maxima`), rounded up to a whole pixel, and the centre is `(row, col)` (columns 1 and 2). Each row of the result is `[row, col, radius]`, as integers.

**Hint:** `maxima[:, 3]` selects a whole column. `np.ceil` rounds up, `np.column_stack` joins columns into one array, and `.astype(int)` converts it to integers.

**Check:** larger circles are drawn on the larger, nearer flowers in the Step 9 figure.

### Step 10 — `detect_blobs(image)`

**Goal:** the whole pipeline run. Steps 5, 8 and 9 are called in order, each result is stored on `self`, and `self.blobs` is returned. The other steps are not called here, because they are already called inside these three.

| Step | Stored on `self` as |
|------|---------------------|
| 5 | `self.laplace_scale_space` |
| 8 | `self.maxima` |
| 9 | `self.blobs` |

**Hint:** `det_coords` in Lab 05 (Step 9) has the same structure.

**Check:** the number of blobs is reported by the evaluate script (about 1400 with the default settings — many are small blobs in the flower texture as well as the flower heads), and the Step 10 figure matches Step 9. The effect of a higher `thresh` can be considered.

## What the Pytests Check

The tests are run with:

```bash
pytest test_lab06.py
```

- **Step 1** — matches the Gaussian formula, as a `(1, k)` array, for two values of `sigma`.
- **Step 2** — matches the second-derivative formula for two values of `sigma`; the kernel is negative at its centre.
- **Step 3** — matches 2D convolution with the equivalent full kernels; vertical stripes give a strong `Lxx` and an almost zero `Lyy`.
- **Step 4** — equals `sigma² · (Lxx + Lyy)` for two values of `sigma`.
- **Step 5** — the shape is `(m, n, num_scales)`, and every layer matches `laplacian()` at that scale.
- **Step 6** — only interior points above the threshold are returned, with indices into the full array; `thresh` is respected.
- **Step 7** — a true peak is accepted with the right sigma; a stronger neighbour in the same layer or in the layer above or below causes rejection; points outside the 3 × 3 × 3 block are ignored.
- **Step 8** — every maximum is found and nothing else; an empty `(0, 4)` array is returned when there are none.
- **Step 9** — the radius is `√2 · sigma` rounded up, and the centre is `(row, col)`, as integers.
- **Step 10** — five synthetic discs of known position and size are all found at the right size; every intermediate is stored on `self`; `thresh` is respected.

## Evaluation Script

In `evaluate_lab06.py` the sunflowers image is downloaded, each step is run in turn, and one figure per completed step is saved to `lab06_results.html`:

- the input image;
- **Steps 1–2** — the two 1D kernels, as line plots;
- **Steps 3–4** — `Lxx`, `Lyy` and the LoG response at one scale;
- **Step 5** — 12 representative scales of the scale space;
- **Step 6** — the number of candidate points at each scale;
- **Step 7** — a small synthetic scale space, with the result of `lap_max_det` in the title;
- **Step 8** — the local maxima as dots on the image;
- **Steps 9–10** — the detected blobs as circles on the image.

Images are shrunk for display (the algorithm itself runs at full resolution) to keep the report small.

The script is run with:

```bash
python evaluate_lab06.py
```

`lab06_results.html` is then opened — in VS Code / Codespaces, by right-clicking the file → **Open with Live Server**, or by running `python -m http.server 5000` and opening the forwarded port.

## Report

`report_lab06.md` is to be filled in.
