#https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix

import numpy as np
import scipy.sparse as ssp

segfault=False

A = ssp.csr_matrix([ [-1.,  1.,  0.,  0.,  0.,  0.,  0.],
                     [ 1., -2.,  1.,  0.,  0.,  0.,  0.],
                     [ 0.,  1., -2.,  1.,  0.,  0.,  0.],
                     [ 0.,  0.,  1., -2.,  1.,  0.,  0.],
                     [ 0.,  0.,  0.,  1., -2.,  1.,  0.],
                     [ 0.,  0.,  0.,  0.,  1., -2.,  1.],
                     [ 0.,  0.,  0.,  0.,  0.,  1., -1.]])

print(A.toarray())


row_idx = 2
new_row_data = np.array([44.,54.,64.])
new_row_indices = np.array([1,2,3])

N_elements_new_row = len(new_row_data)

assert N_elements_new_row == len(new_row_indices)

idx_start_row = A.indptr[row_idx]
idx_end_row = A.indptr[row_idx + 1]

A.data = np.r_[A.data[:idx_start_row], new_row_data, A.data[idx_end_row:]]
A.indices = np.r_[A.indices[:idx_start_row], new_row_indices, A.indices[idx_end_row:]]

if segfault: # this is what was shown on site
    A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + N_elements_new_row]
else: # This is what works
    A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):]]


print(A.toarray()) # segmentation fault when segfault==True
