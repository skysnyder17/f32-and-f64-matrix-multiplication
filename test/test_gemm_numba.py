import numpy as np
import numba as nb

# Import the gemm function
from gemm_numba import gemm

# Define the test function
def test_gemm_numba():
    # Generate random input matrices
    np.random.seed(0)
    A = np.random.randn(100, 50)
    B = np.random.randn(50, 80)
    C = np.random.randn(100, 80)

    # Define the scaling factors
    alpha = 1.5
    beta = 2.0

    # Perform matrix multiplication using gemm function
    C_out = gemm(alpha, A, B, beta, C)

    # Perform matrix multiplication using numpy for comparison
    C_expected = alpha * np.matmul(A, B) + beta * C

    # Check that the output matrix matches the expected output matrix
    assert np.allclose(C_out, C_expected, rtol=1e-4, atol=1e-4)

    print("Test passed!")

# Run the test function
if __name__ == '__main__':
    test_gemm_numba()
