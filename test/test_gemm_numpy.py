import numpy as np

# Import the gemm function
from gemm import gemm

# Define the input matrices A, B, and C
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=np.float32)
C = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

# Define the scaling factors alpha and beta
alpha = 1.0
beta = 2.0

# Call the gemm function
C_out = gemm(alpha, A, B, beta, C)

# Define the expected output matrix
C_expected = np.array([[104, 120, 136], [235, 272, 309], [366, 424, 482]], dtype=np.float32)

# Check that the output matrix matches the expected output matrix
assert np.allclose(C_out, C_expected, rtol=1e-4, atol=1e-4)
