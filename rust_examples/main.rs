
use std::time::{Instant};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use zpr_multithread_matrix::zpr_matrix;


fn main() {
    // Check number of available CPU's (meaning achievable parallel computing)
    let threads = std::thread::available_parallelism().unwrap().get();
    println!("Available CPU's: {:?}", threads);


    // Small matrices - single vs multi-threaded
    let a = zpr_matrix::Matrix::new(
        2,
        2,
        &[1., 2., 3., 4.]
    );

    let b = zpr_matrix::Matrix::new(
        2,
        2,
        &[1., 2., 3., 4.]
    );


    let start_time_1 = Instant::now();
    let c = zpr_matrix::multiply(&a, &b);
    let end_time_1 = Instant::now();
    println!("Small matrices - Single threaded: {:?}", end_time_1-start_time_1);

    // C.print();
    let threads2 = std::thread::available_parallelism().unwrap().get();
    let start_time_2 = Instant::now();
    let d = zpr_matrix::multi_threaded_multiply(&a, &b, threads2);
    let end_time_2 = Instant::now();
    println!("Small matrices - Multi threaded: {:?}", end_time_2-start_time_2);
    println!("Equal?: {:?}", c.compare(&d));

    // Large matrices - single vs multi-threaded
    let mut rng = StdRng::seed_from_u64(42);
    let large_mat_rows = 500;
    let large_mat_cols = 500;
    let large_mat_size = large_mat_rows*large_mat_cols;
    let mut data1 = Vec::with_capacity(large_mat_size);
    let mut data2 = Vec::with_capacity(large_mat_size);
    for _ in 0..large_mat_size {
        data1.push(rng.gen::<f64>());
        data2.push(rng.gen::<f64>());
    }

    let a = zpr_matrix::Matrix::new(
        large_mat_rows,
        large_mat_cols,
        &data1
    );

    let b = zpr_matrix::Matrix::new(
        large_mat_rows,
        large_mat_cols,
        &data2
    );

    let start_time_1 = Instant::now();
    let c = zpr_matrix::multiply(&a, &b);
    let end_time_1 = Instant::now();
    println!("Large matrices - Single threaded: {:?}", end_time_1-start_time_1);

    // C.print();
    let threads2 = std::thread::available_parallelism().unwrap().get();
    let start_time_2 = Instant::now();
    let d = zpr_matrix::multi_threaded_multiply(&a, &b, threads2);
    let end_time_2 = Instant::now();
    println!("Large matrices - Multi threaded: {:?}", end_time_2-start_time_2);
    println!("Equal?: {:?}", c.compare(&d));



}
