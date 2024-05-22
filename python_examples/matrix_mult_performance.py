import numpy as np
import multithread_matrix as mm
import time


def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time

        if duration >= 1:
            unit = "s"
        elif duration >= 1e-3:
            unit = "ms"
            duration *= 1e3
        else:
            unit = "us"
            duration *= 1e6

        print(f"Time taken for {func.__name__}: {duration:.3f} {unit}\n")
        return result

    return wrapper


def generate_random_matrix(size):
    return np.random.rand(size, size).astype(np.float64)


@measure_time
def example_multiplication(matrix_size):
    print(f"Multiplication of {matrix_size}x{matrix_size} Matrices")
    matrix_a = generate_random_matrix(matrix_size)
    matrix_b = generate_random_matrix(matrix_size)
    mm.multiply_f64(matrix_a, matrix_b)


@measure_time
def example_multiplication_single(matrix_size):
    print(f"Multiplication of {matrix_size}x{matrix_size} Matrices (Single-threaded)")
    matrix_a = generate_random_matrix(matrix_size)
    matrix_b = generate_random_matrix(matrix_size)
    mm.multiply_f64(matrix_a, matrix_b, num_threads=1)


if __name__ == "__main__":
    sizes_to_test = [25, 50, 75, 100, 125, 150, 200, 500, 1000]

    for size in sizes_to_test:
        example_multiplication(size)
        example_multiplication_single(size)
