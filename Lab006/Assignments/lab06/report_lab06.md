# Lab Report — Lab 06: Scale Invariant Blob Detector

*Answer each question briefly and factually — a number, a name, or a short phrase is enough.
Each question is marked on whether the required information is present and correct, not on how
it's written, so there is no benefit to writing full sentences or extra explanation.*

**Q1.** In one phrase, what does `detect_blobs(image)` compute?

**Q2.** In Step 2 (`gauss_2nd_deriv_kernel`), the kernel is the Gaussian's derivative of which
order?

**Q3.** Which library function did you use in Step 3 (`second_derivatives`) for each 1D
convolution?

**Q4.** What does Step 4 (`laplacian`) multiply `(Lxx + Lyy)` by before returning it?

**Q5.** Which NumPy function did you use in Step 5 (`build_scale_space`) to stack the layers into
a 3D array?

**Q6.** What is the default value of `self.thresh` in the constructor?

**Q7.** In Step 6 (`candidate_points`), what number is added to the indices returned by
`np.argwhere`?

**Q8.** In Step 7 (`lap_max_det`), how many values (including the point itself) are in the block
the point is compared against?

**Q9.** In Step 9 (`blob_circles`), what is the blob radius, as a multiple of sigma?

**Q10.** What order do the four values in each row of `find_maxima`'s output array come in?
