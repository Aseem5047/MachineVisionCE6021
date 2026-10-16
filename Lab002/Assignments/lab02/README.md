# Lab 02 — MRI Noise Removal via Fourier Notch Filtering

## Introduction

Magnetic Resonance Imaging (MRI) data is acquired in the *frequency domain* (called **k-space**) and transformed back to the spatial domain for viewing. Periodic noise — such as electrical interference — appears as bright, isolated spikes in k-space, and as a stripe pattern across the reconstructed image.

In this lab you will find the noise spikes in a corrupted MRI scan and remove them with a **notch filter**, restoring a clean image. The lab is split into five small steps, and you can check each one before moving on to the next.

## Dataset / Test Images

| Resource | Details |
|----------|---------|
| `corrupted_mrimage1d.txt` | The corrupted scan's k-space data (256 × 256 complex values, stored flattened). Located in `Data/`. Loaded for you by `evaluate_lab02.py`. |
| `corrupted_image.png` | Reference image of the corrupted scan — used by the tests. Located in `Data/`. |
| `filtered_image.png` | Reference image of the correctly filtered scan — used by the tests. Located in `Data/`. |

## How This Lab Works

1. Implement **one step** in `lab02.py`.
2. Run `python evaluate_lab02.py`. It runs every step that is finished, adds a figure for each one to `lab02_results.html`, and stops at the first step that isn't done yet.
3. Open `lab02_results.html`, check the new figure looks right, then move on to the next step.

You can also run `pytest test_lab02.py` at any point — the tests are grouped by step, so the failing tests tell you which step needs attention.

> **Positions are always written as `(row, col)`** in this lab — row first (up/down), column second (left/right). Mixing these up is the most common source of bugs.

## Background: k-space and the Centred Spectrum

This lab builds on lesson **01_07 — 2D Fourier transform**.

The 2D Fourier transform turns an image into a grid of complex numbers, one per spatial frequency. By convention the result is **shifted** (with `np.fft.fftshift`) so that the lowest frequencies sit in the **centre** of the grid and higher frequencies sit further out. The data you are given, `f_shifted`, is already in this centred form.

- The values are complex, so to *look* at them we plot their magnitude, `np.abs(...)`.
- The magnitudes vary enormously (the centre is far brighter than everything else), so we plot `np.log(np.abs(...) + 1)` to make the rest visible. The `+ 1` avoids taking `log(0)`.
- To get an image back, the shift is undone with `np.fft.ifftshift`, then the inverse transform `np.fft.ifft2` is applied.

## Background: Why Periodic Noise Makes Two Spikes

A sinusoidal stripe pattern contains just **one** spatial frequency, so in the spectrum it shows up as a single bright point — plus its **mirror image** on the opposite side of the centre. (The spectrum of any real-valued image is symmetric through the centre, so every frequency appears twice.)

That's why you will see **two** isolated spikes, placed symmetrically either side of the centre. Both must be removed: if one spike is at `(128 + a, 128 + b)`, the other is at `(128 − a, 128 − b)`.

## Background: The Notch Filter

A **notch filter** removes a small region of the spectrum and leaves everything else untouched. Here it is a **mask** — an array the same shape as the spectrum, full of `1`s, with a small disc of `0`s around each noise spike. Multiplying the spectrum by the mask zeroes out the noise frequencies; reconstructing the image from the masked spectrum gives the cleaned scan.

## Your Task

### Class: `MRIprocessor`

**Constructor:** `MRIprocessor(shape=(256, 256))` — `shape` is the image size as `(rows, cols)`, available inside every method as `self.shape`. Instantiating the class is already done for you in `evaluate_lab02.py`.

### Methods to Implement

| Step | Method | Returns |
|------|--------|---------|
| 1 | `fourier_magnitude(f_shifted)` | 2D float array: the log-magnitude spectrum |
| 2 | `reconstruct(f_shifted)` | 2D float array: the spatial image |
| 3 | `get_noise_peaks()` | list of two `(row, col)` tuples |
| 4 | `make_notch_mask(centres, radius=2)` | 2D float array of 0s and 1s, shape `self.shape` |
| 5 | `remove_noise(f_shifted)` | tuple `(filtered_img, corrupted_img, fourier_img, mask)` |

## Step-by-Step Walkthrough

### Step 1 — `fourier_magnitude(f_shifted)`

**Goal:** an image of the spectrum you can look at.

Return the log-magnitude of the data: take the absolute value with `np.abs`, add `1`, then take `np.log`.

**Check:** run `evaluate_lab02.py`. The first figure shows a bright centre fading outwards.

### Step 2 — `reconstruct(f_shifted)`

**Goal:** turn centred k-space data back into a viewable image.

Three operations, in order:
1. `np.fft.ifftshift` — undo the centring shift.
2. `np.fft.ifft2` — the inverse 2D Fourier transform.
3. `np.fft.fftshift` on the result — this moves the scan into the middle of the image (without it, the image appears split into four quarters).

Finally take `np.abs(...)`, because the inverse transform returns complex numbers and you want a real-valued image.

```python
image = np.fft.ifft2(np.fft.ifftshift(f_shifted))
return np.abs(np.fft.fftshift(image))
```

**Check:** the second figure shows the MRI scan, centred, with a stripe pattern across it.

### Step 3 — `get_noise_peaks()` (find them graphically)

**Goal:** the `(row, col)` positions of the two noise spikes.

1. With steps 1 and 2 done, run `evaluate_lab02.py` and open `lab02_results.html`.
2. In the **Step 1** spectrum figure, **hover your mouse** over the image — a label shows the `row` and `col` under the cursor. You can also drag a box to zoom in, and double-click to zoom back out.
3. Ignore the large bright region in the centre — that's the ordinary content of the scan. Look for **two small, isolated bright dots** placed symmetrically either side of the centre.
4. Write their positions down as `(row, col)` and return them as a list:
   ```python
   return [(row1, col1), (row2, col2)]
   ```

**Check:** the two positions should be mirror images through the centre `(128, 128)` — see *Why Periodic Noise Makes Two Spikes* above. Being off by a pixel is fine.

### Step 4 — `make_notch_mask(centres, radius=2)`

**Goal:** an array of `1`s with a disc of `0`s of the given `radius` around each `(row, col)` in `centres`.

1. Start with an array of ones the same shape as the image:
   ```python
   rows, cols = self.shape
   mask = np.ones((rows, cols))
   ```
2. Build two arrays holding each pixel's row and column number. `np.ogrid` does this efficiently:
   ```python
   R, C = np.ogrid[:rows, :cols]
   ```
3. For each centre, compute every pixel's distance from it, and set the pixels within `radius` to zero:
   ```python
   for r, c in centres:
       dist = np.sqrt((R - r) ** 2 + (C - c) ** 2)
       mask[dist <= radius] = 0
   ```
4. Return `mask`.

**Check:** the Step 4 figure shows a white image with two small black dots exactly where your spikes were.

### Step 5 — `remove_noise(f_shifted)`

**Goal:** put the four steps together. Call your own methods — don't repeat their code here.

1. `fourier_img` — the spectrum, from `self.fourier_magnitude(f_shifted)`.
2. `corrupted_img` — the image before filtering, from `self.reconstruct(f_shifted)`.
3. `mask` — from `self.make_notch_mask(self.get_noise_peaks())`.
4. `filtered_img` — multiply the spectrum by the mask, then reconstruct: `self.reconstruct(f_shifted * mask)`.
5. Return all four, **in this order**:
   ```python
   return filtered_img, corrupted_img, fourier_img, mask
   ```

**Check:** the final 2×2 figure shows the stripes are gone from the filtered image.

## What the Pytests Check

Run the tests with:

```bash
pytest test_lab02.py
```

- **Step 1** — the magnitude is real-valued and equals `log(|f_shifted| + 1)`.
- **Step 2** — the reconstruction is a real float image of shape (256, 256) that matches the corrupted reference (a failure here usually means the final `fftshift` is missing).
- **Step 3** — each noise spike is within 2 pixels of one of your positions.
- **Step 4** — the mask has the right shape, contains only 0s and 1s, is zero at each centre and one far away, has roughly the right disc area, and gets bigger as `radius` increases.
- **Step 5** — `remove_noise` returns four outputs; the filtered image closely matches the clean reference (SSIM ≥ 0.85) and no longer resembles the corrupted one (SSIM < 0.50).

## Evaluation Script

`evaluate_lab02.py` loads the k-space data, runs each step in turn, and saves one figure per completed step to `lab02_results.html`:

- **Step 1** — the spectrum, with `(row, col)` shown when you hover (used to find the spikes).
- **Step 2** — the reconstructed corrupted image.
- **Step 4** — the notch mask.
- **Step 5** — a 2×2 grid: corrupted image, spectrum, mask, filtered image.

Run it with:

```bash
python evaluate_lab02.py
```

Then open `lab02_results.html` — in VS Code / Codespaces, right-click the file → **Open with Live Server**, or run `python -m http.server 5000` and open the forwarded port.

## Report

When you're done, fill in `report_lab02.md`.
