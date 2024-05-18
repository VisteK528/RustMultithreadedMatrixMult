#[path = "../src/lib.rs"] mod main;
use main::zpr_matrix;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn my_test() {
        let A = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{1., 2., 3., 4.}
        );

        let B = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{1., 2., 3., 4.}
        );

        let C_expect = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{7., 10., 15., 22.}
        );

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &B);

        assert_eq!(C_expect.compare(&C), true);
    }

    #[test]
    fn test_non_square_matrices() {
        let A = zpr_matrix::Matrix::new(
            2,
            3,
            &vec!{1., 2., 3., 4., 5., 6.}
        );

        let B = zpr_matrix::Matrix::new(
            3,
            2,
            &vec!{7., 8., 9., 10., 11., 12.}
        );

        let C_expect = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{58., 64., 139., 154.}
        );

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &B);

        assert_eq!(C_expect.compare(&C), true);
    }

    #[test]
    fn test_identity_matrix() {
        let A = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9.}
        );

        let I = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{1., 0., 0., 0., 1., 0., 0., 0., 1.}
        );

        let C_expect = A.clone();

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &I);

        assert_eq!(C_expect.compare(&C), true);
    }

    #[test]
    fn test_negative_values() {
        let A = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{1., -2., -3., 4.}
        );

        let B = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{-1., 2., 3., -4.}
        );

        let C_expect = zpr_matrix::Matrix::new(
            2,
            2,
            &vec!{-7., 10., 15., -22.}
        );

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &B);

        assert_eq!(C_expect.compare(&C), true);
    }

    #[test]
    fn test_rectangular_matrices() {
        let A = zpr_matrix::Matrix::new(
            3,
            2,
            &vec!{1., 2., 3., 4., 5., 6.}
        );

        let B = zpr_matrix::Matrix::new(
            2,
            3,
            &vec!{7., 8., 9., 10., 11., 12.}
        );

        let C_expect = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{27., 30., 33., 61., 68., 75., 95., 106., 117.}
        );

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &B);

        assert_eq!(C_expect.compare(&C), true);
    }

    #[test]
    fn test_large_matrices() {
        let A = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9.}
        );

        let B = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{9., 8., 7., 6., 5., 4., 3., 2., 1.}
        );

        let C_expect = zpr_matrix::Matrix::new(
            3,
            3,
            &vec!{30., 24., 18., 84., 69., 54., 138., 114., 90.}
        );

        let C = zpr_matrix::multi_threaded_multiply_channel(&A, &B);

        assert_eq!(C_expect.compare(&C), true);
    }
}