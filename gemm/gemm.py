import numpy as np

def gemm(alpha: float, A: np.ndarray, B: np.ndarray, beta: float, C: np.ndarray) -> np.ndarray:
    """
    General matrix multiplication for f32 or f64 matrices with arbitrary row and column stride.
    Calculates the product of two matrices, A and B, with scaling factors alpha and beta, and adds the result
    to a third matrix C.
    
    Parameters:
    alpha (float): Scaling factor for the product of A and B.
    A (np.ndarray): Input matrix A with shape (M, K) and data type f32 or f64.
    B (np.ndarray): Input matrix B with shape (K, N) and data type f32 or f64.
    beta (float): Scaling factor for matrix C.
    C (np.ndarray): Input/output matrix C with shape (M, N) and data type f32 or f64.
    
    Returns:
    np.ndarray: The output matrix C with the result of the matrix multiplication.
    """
    # Get the data types of the input matrices
    dtype_A, dtype_B, dtype_C = A.dtype, B.dtype, C.dtype

    # Convert the input matrices to a common data type
    common_dtype = np.promote_types(dtype_A, dtype_B, dtype_C)
    A = A.astype(common_dtype)
    B = B.astype(common_dtype)
    C = C.astype(common_dtype)

    # Check the shapes of the input matrices
    assert A.shape[1] == B.shape[0]
    assert A.shape[0] == C.shape[0]
    assert B.shape[1] == C.shape[1]

    # Call the BLAS function for general matrix multiplication
    C = alpha * np.matmul(A, B) + beta * C

    # Convert the output matrix to the original data type
    C = C.astype(dtype_C)

    return C
