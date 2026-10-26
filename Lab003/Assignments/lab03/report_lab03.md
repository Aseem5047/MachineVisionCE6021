# Lab Report — Lab 03: Fluid Level Detection via the Hough Transform

*Answer each question briefly and factually — a number, a name, or a short phrase is enough.
Each question is marked on whether the required information is present and correct, not on how
it's written, so there is no benefit to writing full sentences or extra explanation.*

**Q1.** In one phrase, what does `detect_fluid_levels()` compute?

The fluid level (y pixel position) in each of the three bottles — `[187.5, 108.0, 190.5]`.

**Q2.** Which OpenCV function did you use in Step 1 (`threshold`) to binarise the image?

`cv2.threshold` with `cv2.THRESH_BINARY_INV` (threshold value 240).

**Q3.** What `kernel_size` did you use for the Step 2 (`close_gaps`) morphological closing?

5 (a 5×5 square kernel, `self.kernel_size`).

**Q4.** What `canny_low` / `canny_high` values did you use for Step 3 (`detect_edges`)?

50 / 150.

**Q5.** In one phrase, what does the Step 4 (`mask_roi`) region-of-interest mask remove from the
edge image?

All edges outside the region of interest (rows 80–200, the bottle necks) — e.g. bottle bases, labels and background.

**Q6.** Which OpenCV function does Step 5 (`find_lines`) use to find line segments?

`cv2.HoughLinesP` (probabilistic Hough transform).

**Q7.** In Step 6 (`select_horizontal_lines`), how is a line's slope calculated from its endpoints
`(x1, y1, x2, y2)`?

slope = (y2 − y1) / (x2 − x1); vertical lines (x1 == x2) are skipped, and a line is kept if |slope| ≤ 0.1.