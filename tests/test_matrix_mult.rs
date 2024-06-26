#[path = "../src/lib.rs"] mod main;
use main::matrix;

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn my_test() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            2,
            2,
            &vec!{1., 2., 3., 4.}
        );

        let b = matrix::Matrix::new(
            2,
            2,
            &vec!{1., 2., 3., 4.}
        );

        let c_expect = matrix::Matrix::new(
            2,
            2,
            &vec!{7., 10., 15., 22.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_non_square_matrices() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            2,
            3,
            &vec!{1., 2., 3., 4., 5., 6.}
        );

        let b = matrix::Matrix::new(
            3,
            2,
            &vec!{7., 8., 9., 10., 11., 12.}
        );

        let c_expect = matrix::Matrix::new(
            2,
            2,
            &vec!{58., 64., 139., 154.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_identity_matrix() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            3,
            3,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9.}
        );

        let i = matrix::Matrix::new(
            3,
            3,
            &vec!{1., 0., 0., 0., 1., 0., 0., 0., 1.}
        );

        let c_expect = a.clone();

        let c = matrix::multi_threaded_multiply(&a, &i, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_negative_values() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            2,
            2,
            &vec!{1., -2., -3., 4.}
        );

        let b = matrix::Matrix::new(
            2,
            2,
            &vec!{-1., 2., 3., -4.}
        );

        let c_expect = matrix::Matrix::new(
            2,
            2,
            &vec!{-7., 10., 15., -22.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_rectangular_matrices() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            3,
            2,
            &vec!{1., 2., 3., 4., 5., 6.}
        );

        let b = matrix::Matrix::new(
            2,
            3,
            &vec!{7., 8., 9., 10., 11., 12.}
        );

        let c_expect = matrix::Matrix::new(
            3,
            3,
            &vec!{27., 30., 33., 61., 68., 75., 95., 106., 117.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_large_matrices() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            3,
            3,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9.}
        );

        let b = matrix::Matrix::new(
            3,
            3,
            &vec!{9., 8., 7., 6., 5., 4., 3., 2., 1.}
        );

        let c_expect = matrix::Matrix::new(
            3,
            3,
            &vec!{30., 24., 18., 84., 69., 54., 138., 114., 90.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }

    #[test]
    fn test_5x5_matrices() {
        let threads: usize = std::thread::available_parallelism().unwrap().get();

        let a = matrix::Matrix::new(
            5,
            5,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.}
        );

        let b = matrix::Matrix::new(
            5,
            5,
            &vec!{1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.}
        );

        let c_expect = matrix::Matrix::new(
            5,
            5,
            &vec!{215., 230., 245., 260., 275., 490., 530., 570., 610., 650., 765., 830., 895., 960., 1025., 1040., 1130., 1220., 1310., 1400., 1315., 1430., 1545., 1660. , 1775.}
        );

        let c = matrix::multi_threaded_multiply(&a, &b, threads);

        assert_eq!(c_expect.compare(&c), true);
    }
}