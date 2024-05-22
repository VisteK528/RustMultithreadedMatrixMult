import numpy as np
import multithread_matrix
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        if duration >= 1:
            unit = "s"
            duration *= 1
        elif duration >= 1e-3:
            unit = "ms"
            duration *= 1e3
        else:
            unit = "us"
            duration *= 1e6

        print(f"Time taken for {func.__name__}: {duration:.3f} {unit}\n")
        return result

    return wrapper


@measure_time
def example_basic_multiplication():
    print("Example 1: Basic Multiplication of 2x2 Matrices")
    multithread_matrix.multiply_f64(
        np.array([[1, 2], [3, 4]], dtype=np.float64),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
    )


@measure_time
def example_3x3_multiplication():
    print("Example 2: Multiplication of 3x3 Matrices")
    multithread_matrix.multiply_f64(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=np.float64),
    )


@measure_time
def example_3x3_multiplication_single():
    print("Example 2: Multiplication of 3x3 Matrices")
    multithread_matrix.multiply_f64(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64),
        np.array([[9.0, 8.0, 7.0], [6.0, 5.0, 4.0], [3.0, 2.0, 1.0]], dtype=np.float64),
        num_threads=1,
    )


@measure_time
def example_non_square_multiplication():
    print("Example 3: Multiplication of Non-Square Matrices (2x3 and 3x2)")
    multithread_matrix.multiply_f64(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64),
        np.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], dtype=np.float64),
    )


@measure_time
def example_identity_matrix_multiplication():
    print("Example 4: Multiplication with Identity Matrix")
    identity_matrix = np.eye(3, dtype=np.float64)
    multithread_matrix.multiply_f64(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), identity_matrix
    )


@measure_time
def example_large_matrices_multiplication():
    print("Example 5: Large Matrices")
    matrix_a = np.random.rand(100, 200).astype(np.float64)
    matrix_b = np.random.rand(200, 100).astype(np.float64)
    multithread_matrix.multiply_f64(matrix_a, matrix_b)

@measure_time
def example_zero_matrix_multiplication():
    print("Example 6: Multiplication with Zero Matrix")
    zero_matrix = np.zeros((2, 2), dtype=np.float64)
    multithread_matrix.multiply_f64(
        np.array([[1, 2], [3, 4]], dtype=np.float64), zero_matrix
    )


@measure_time
def example_float_integer_multiplication():
    print("Example 7: Multiplication of Float and Integer Matrices")
    multithread_matrix.multiply_f64(
        np.array([[1, 2], [3, 4]], dtype=np.float64),
        np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
    )


@measure_time
def example_large_values_multiplication():
    print("Example 8: Handling Large Values")
    large_values_matrix = np.array([[1e10, 2e10], [3e10, 4e10]], dtype=np.float64)
    multithread_matrix.multiply_f64(
        large_values_matrix, np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    )


@measure_time
def example_transposed_matrix_multiplication():
    print("Example 9: Multiplication with Transposed Matrix")
    matrix_a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    matrix_b = matrix_a.T
    multithread_matrix.multiply_f64(matrix_a, matrix_b)


@measure_time
def simple_mult():
    ("Example 10: Basic Multiplication of 2x2 Matrices")
    multithread_matrix.multiply_f64(
        np.array([[1, 2], [3, 4]], dtype=np.float64),
        np.array([[1, 2], [3, 4]], dtype=np.float64),
    )


@measure_time
def example_large_matrix_multiplication():
    print("Example: Large Matrix Multiplication")
    matrix_size = 50
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    multithread_matrix.multiply_f64(matrix_a, matrix_b)



@measure_time
def example_large_matrix_multiplication_single():
    print("Example: Large Matrix Multiplication")
    matrix_size = 50
    matrix_a = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    matrix_b = np.random.rand(matrix_size, matrix_size).astype(np.float64)
    multithread_matrix.multiply_f64(matrix_a, matrix_b, num_threads=1)


if __name__ == "__main__":
    example_basic_multiplication()
    example_non_square_multiplication()
    example_identity_matrix_multiplication()
    example_large_matrices_multiplication()
    example_zero_matrix_multiplication()
    example_float_integer_multiplication()
    example_large_values_multiplication()
    example_transposed_matrix_multiplication()
    simple_mult()
    example_3x3_multiplication_single()
    example_3x3_multiplication()
    example_large_matrix_multiplication()
    example_large_matrix_multiplication_single()