#[path = "../src/lib.rs"] mod main;
use main::matrix;

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_matrix_data(rows: usize, cols: usize) -> Vec<f64> {
        (0..rows * cols).map(|x| x as f64).collect()
    }

    fn matrix_multiplication_test(rows: usize, cols: usize, threads: usize) {
        let data_a = generate_matrix_data(rows, cols);
        let data_b = generate_matrix_data(cols, rows);
        let matrix_a = matrix::Matrix::new(rows, cols, &data_a);
        let matrix_b = matrix::Matrix::new(cols, rows, &data_b);
        let single_threaded_result = matrix::multi_threaded_multiply(&matrix_a, &matrix_b, 1);
        let multi_threaded_result = matrix::multi_threaded_multiply(&matrix_a, &matrix_b, threads);

        assert_eq!(single_threaded_result.shape(), multi_threaded_result.shape());
        assert_eq!(single_threaded_result.data(), multi_threaded_result.data());
    }

    #[test]
    fn test_small_matrix_multiplication() {
        matrix_multiplication_test(2, 2, 1);
        matrix_multiplication_test(2, 2, 2);
        matrix_multiplication_test(2, 2, 4);
    }

    #[test]
    fn test_medium_matrix_multiplication() {
        matrix_multiplication_test(10, 10, 1);
        matrix_multiplication_test(10, 10, 2);
        matrix_multiplication_test(10, 10, 4);
        matrix_multiplication_test(10, 10, 8);
    }

    #[test]
    fn test_large_matrix_multiplication() {
        matrix_multiplication_test(100, 100, 1);
        matrix_multiplication_test(100, 100, 2);
        matrix_multiplication_test(100, 100, 4);
        matrix_multiplication_test(100, 100, 8);
        matrix_multiplication_test(100, 100, 16);
    }

    #[test]
    fn test_different_thread_counts_and_sizes() {
        let sizes = vec![(2, 2), (10, 10), (100, 100)];
        let thread_counts = vec![1, 2, 4, 8, 16];

        for (rows, cols) in sizes.iter() {
            for threads in thread_counts.iter() {
                matrix_multiplication_test(*rows, *cols, *threads);
            }
        }
    }
}
