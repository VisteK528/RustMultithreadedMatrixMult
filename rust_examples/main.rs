
use std::time::{Instant};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use multithread_matrix::matrix;


fn main() {
    let mut rng = StdRng::seed_from_u64(42);

    // Check number of available CPU's (meaning achievable parallel computing)
    let threads = std::thread::available_parallelism().unwrap().get();
    println!("Available CPU's: {:?}", threads);
    let sizes = vec![2usize, 25usize, 100usize, 1000usize, 2000usize];
    let names = vec!["Tiny", "Small", "Medium", "Large", "Enormous"];

    for tuple in sizes.iter().zip(names.iter()){
        let (size, name) = tuple;

        // Medium matrices - single vs multithreaded
        let mat_rows = *size;
        let mat_cols = *size;
        let mat_size = mat_rows*mat_rows;
        let mut data1 = Vec::with_capacity(mat_size);
        let mut data2 = Vec::with_capacity(mat_size);
        for _ in 0..mat_size {
            data1.push(rng.gen::<f64>());
            data2.push(rng.gen::<f64>());
        }

        let a = matrix::Matrix::new(
            mat_rows,
            mat_cols,
            &data1
        );

        let b = matrix::Matrix::new(
            mat_rows,
            mat_cols,
            &data2
        );

        let start_time_1 = Instant::now();
        let c = matrix::multi_threaded_multiply(&a, &b, 1);
        let end_time_1 = Instant::now();
        println!("\n{:?} matrices {:?}x{:?}", name, a.shape().0, a.shape().1);
        println!("Single threaded: {:?}", end_time_1-start_time_1);

        // C.print();
        let threads2 = std::thread::available_parallelism().unwrap().get();
        let start_time_2 = Instant::now();
        let d = matrix::multi_threaded_multiply(&a, &b, threads2);
        let end_time_2 = Instant::now();
        println!("Multi threaded: {:?}", end_time_2-start_time_2);
        assert_eq!(c.compare(&d), true);
    }
}
