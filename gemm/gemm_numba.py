import numba as nb
import numpy as np

@nb.njit(parallel=True)
def gemm(alpha, A, B, beta, C):
    """
    General matrix multiplication for f32, f64 matrices. Operates on matrices with general layout
    (they can use arbitrary row and column stride).
    
    Computes C = alpha * A @ B + beta * C where A is an m-by-k matrix, B is a k-by-n matrix,
    and C is an m-by-n matrix.
    
    Parameters:
    -----------
    alpha: float
        Scaling factor for A @ B.
    A: ndarray
        The first input matrix, an m-by-k matrix.
    B: ndarray
        The second input matrix, a k-by-n matrix.
    beta: float
        Scaling factor for C.
    C: ndarray
        The output matrix, an m-by-n matrix.
    
    Returns:
    --------
    ndarray
        The result of the matrix multiplication, an m-by-n matrix.
    """
    
    # Ensure that the input matrices have compatible shapes
    assert A.shape[1] == B.shape[0], "Incompatible matrix shapes"
    assert C.shape[0] == A.shape[0] and C.shape[1] == B.shape[1], "Incompatible matrix shapes"
    
    # Compute the matrix multiplication
    m, k = A.shape
    k, n = B.shape
    
    C_out = np.empty_like(C)
    
    for i in nb.prange(m):
        for j in range(n):
            dot_product = 0.0
            for l in range(k):
                dot_product += A[i, l] * B[l, j]
            C_out[i, j] = alpha * dot_product + beta * C[i, j]
    
    return C_out
