# f32-and-f64-matrix-multiplication
general matrix multiplication function for f32 and f64 matrices.


### `gemm.py`

This script implements a general matrix multiplication function in Python for f32 and f64 matrices. It operates on matrices with general layout and can use arbitrary row and column stride.

The `gemm` function takes in a scaling factor `alpha`, input matrices `A` and `B`, a scaling factor `beta`, and an output matrix `C`. It performs a matrix multiplication of `A` and `B`, scaled by `alpha`, and adds the result to `C`, scaled by `beta`. The resulting matrix is returned.

This implementation uses the `numpy` library's `matmul` function to perform the matrix multiplication.

### `test_gemm_numpy.py`

This script provides a basic test case for the `gemm` function implemented in `gemm.py`. It generates random matrices using the `numpy` library and compares the output of the `gemm` function to the result of a direct matrix multiplication using the `numpy` library's `matmul` function.

### `test_gemm_numba.py`

This script provides another test case for the `gemm` function implemented in `gemm.py`, this time using the `numba` library to JIT-compile the function for faster execution. It generates random matrices using the `numpy` library and compares the output of the `gemm` function to the result of a direct matrix multiplication using the `numpy` library's `matmul` function.

### `gemm_numba.py`

This script provides an alternative implementation of the `gemm` function in Python using the `numba` library to JIT-compile the function for faster execution. It uses nested loops to perform the matrix multiplication, which is more efficient than using the `numpy` library's `matmul` function for smaller matrices.

This implementation uses the `numba.njit` decorator to JIT-compile the `gemm` function and also uses `numba.prange` to parallelize the outer loop for faster execution on multi-core CPUs.

Overall, these scripts provide a basic implementation of a general matrix multiplication function in Python, along with test cases for validating the correctness of the implementation.
