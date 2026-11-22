# Lab Report — Lab 05: Harris Corner Detector

*Answer each question briefly and factually — a number, a name, or a short phrase is enough.
Each question is marked on whether the required information is present and correct, not on how
it's written, so there is no benefit to writing full sentences or extra explanation.*

**Q1.** In one phrase, what does `det_coords()` compute?

An (N, 2) array of (row, col) coordinates, one per detected corner (81 for the checkerboard).

**Q2.** Which library function did you use in Step 2 (`compute_gradients`) to convolve the image
with the Sobel kernels?

`scipy.signal.convolve2d` (with `mode='same'`, `boundary='symm'`)

**Q3.** In Step 3 (`gradient_products`), what are the three products computed from `Ix` and `Iy`?

`Ix*Ix`, `Iy*Iy` and `Ix*Iy`

**Q4.** What does Step 4 (`smooth_products`) use to weight each gradient product before summing
it over a local window?

A 2D Gaussian kernel (`gaussian_kernel()`, 13×13 for sigma = 2, not normalised)

**Q5.** State the formula used in Step 5 (`det_and_trace`) to compute the determinant from `A`,
`B` and `C`.

`Det(M) = A*B - C**2` (trace = `A + B`)

**Q6.** State the formula used in Step 6 (`harris_response`) to compute the response `R` from the
determinant and trace.

`R = Det(M) - alpha * Trace(M)**2`

**Q7.** What is the default value of `self.alpha` in the constructor?

0.05

**Q8.** What is the default value of `self.threshold_val` in the constructor?

0.001

**Q9.** Which `skimage` function does Step 8 (`find_peaks`) use to collapse a cluster of
high-response pixels down to a single coordinate?

`skimage.feature.peak_local_max` (with `min_distance=self.min_distance`, default 5)

**Q10.** In Step 9 (`det_coords`), what coordinate order — `(row, col)` or `(col, row)` — does the
returned array use?

`(row, col)` — i.e. (y, x); cv2.circle needs `(col, row)`