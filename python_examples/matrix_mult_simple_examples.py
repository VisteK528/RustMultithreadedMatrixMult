import numpy as np
import multithread_matrix as mm

RTOL = 1e-8
ATOL = 1e-8


def name(func):
    def wrapper(*args, **kwargs):
        print(f"\n{func.__name__}")
        result = func(*args, **kwargs)
        return result

    return wrapper


@name
def small_matrix_mult():
    x = np.array([[1, 2], [1, 2]], dtype=np.float64)
    y = np.array([[1, 2], [1, 2]], dtype=np.float64)
    z = mm.multiply_f64(x, y)
    print(z)
    assert np.isclose(z, x @ y, RTOL, ATOL).all() == True


@name
def non_square_matrix_mult():
    x = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.float64)
    y = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    z = mm.multiply_f64(x, y)
    print(z)
    assert np.isclose(z, x @ y, RTOL, ATOL).all() == True


@name
def multiply_by_identity_matrix():
    x = np.array([[1, 2], [3, 4]], dtype=np.float64)
    y = np.array([[1, 0], [0, 1]], dtype=np.float64)
    z = mm.multiply_f64(x, y)
    print(z)
    assert np.isclose(z, x, RTOL, ATOL).all() == True


@name
def multiply_vector_by_vector():
    x = np.array([[1, 2, 3, 4]], dtype=np.float64)
    y = np.array([[1, 0, 1, 0]], dtype=np.float64).T
    z = mm.multiply_f64(x, y)
    print(z)
    assert np.isclose(z, np.array([[4]]), RTOL, ATOL).all() == True


@name
def multiply_matrix_by_vector():
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64)
    y = np.array([[1, 0, 1]], dtype=np.float64).T
    z = mm.multiply_f64(x, y)
    print(z)
    assert np.isclose(z, np.array([[4, 10, 16]]).T, RTOL, ATOL).all() == True


if __name__ == "__main__":
    small_matrix_mult()
    non_square_matrix_mult()
    multiply_by_identity_matrix()
    multiply_vector_by_vector()
    multiply_matrix_by_vector()
