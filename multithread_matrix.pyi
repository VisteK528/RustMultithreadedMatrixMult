import numpy as np

def multiply_f64(a: np.ndarray, b: np.ndarray, num_threads=None) -> np.ndarray:
    """
    Multiplies two matrices of type float64 using multiple threads for parallel computation.

    :param a: First matrix, a 2D NumPy array of type float64.
    :param b: Second matrix, a 2D NumPy array of type float64.
    :param num_threads: Optional; The number of threads to use for the computation.
                        If None, the function will use that number of threads provided by the hardware CPU.
    :return: The result of the matrix multiplication as a 2D NumPy array of type float64.
    :raises ValueError: If the number of columns in the first matrix does not match the number of rows in the second matrix.
    :raises TypeError: If the input arrays are not of type float64.
    """
    pass
