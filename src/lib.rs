use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};

#[allow(dead_code)]
pub mod zpr_matrix{
    use super::*;



    pub struct Matrix{
        rows: usize,
        columns: usize,
        data: Vec<Vec<f64>>
    }

    #[derive(Clone)]
    struct MatrixElement{
        row: usize,
        column: usize,
        value: f64
    }

    impl Clone for Matrix{
        fn clone(&self) -> Matrix{
            let mut matrix_vec = Vec::with_capacity(self.rows*self.columns);
            for row in self.data.clone(){
                for element in row{
                    matrix_vec.push(element);
                }
            }

            Matrix::new(self.rows, self.columns, &matrix_vec)
        }
    }

    impl Matrix{
        pub fn new(rows: usize, columns: usize, data: &[f64]) -> Self {
            assert_eq!(rows*columns, data.len());

            let mut matrix_data = Vec::with_capacity(rows);
            for i in 0..rows{
                let mut row_data = Vec::with_capacity(columns);
                for j in 0..columns{
                    row_data.push(data[i*columns+j]);
                }
                matrix_data.push(row_data);
            }
            Matrix { rows, columns, data: matrix_data }
        }

        pub fn compare(&self, other: &Matrix) -> bool{
            if self.rows == other.rows && self.columns == other.columns && self.data == other.data{
                return true;
            }
            false
        }

        pub fn print(&self) {
            for row in &self.data {
                for &val in row {
                    print!("{:.2} ", val);
                }
                println!();
            }
        }

    }

    pub fn multiply(a: &Matrix, b: &Matrix) -> Matrix{
        // Check if matrices can be multiplied
        assert_eq!(a.columns, b.rows);

        let c_rows = a.rows;
        let c_cols = b.columns;
        let mut c_data: Vec<f64> = vec![0.; c_cols*c_rows];

        for i in 0..c_rows{
            for j in 0..c_cols{
                let mut sum = 0.;
                for k in 0..a.columns{
                    sum += a.data[i][k] * b.data[k][j];
                }
                c_data[i*c_cols+j] = sum;
            }
        }
        Matrix::new(c_rows, c_cols, &c_data)
    }

    pub fn multiply_thread_element(a: &Matrix, b: &Matrix, c_data_mut: &Arc<Mutex<&mut Vec<f64>>>, row: usize, column: usize){
        let mut sum: f64 = 0.;
        for i in 0..a.columns{
            sum += a.data[row][i]*b.data[i][column];
        }
        let mut x = c_data_mut.lock().unwrap();
        x[row*b.columns+column] = sum;
    }

    pub fn multi_threaded_multiply(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.columns, b.rows);

        let c_rows = a.rows;
        let c_cols = b.columns;
        let lock = Arc::new(Mutex::new(vec![0.; c_cols * c_rows]));

        std::thread::scope(|s|{
            let element_thread_handle = |i: usize, j: usize| {
                let mut sum: f64 = 0.;
                for k in 0..a.columns{
                    sum += a.data[i][k]*b.data[k][j];
                }
                let mut x = lock.lock().unwrap();
                x[i*b.columns+j] = sum;


            };

            let mut handles: Vec<std::thread::ScopedJoinHandle<()>> = vec![];
            for i in 0..c_rows {
                for j in 0..c_cols {
                    handles.push(s.spawn(move || {element_thread_handle(i, j);}));


                }
            }

            for handle in handles {
                handle.join().unwrap();
            }


        });



        let x =  Matrix::new(c_rows, c_cols, &lock.lock().unwrap());
        x
    }

    pub fn multi_threaded_multiply_channel(a: &Matrix, b: &Matrix) -> Matrix {
        assert_eq!(a.columns, b.rows);

        let c_rows = a.rows;
        let c_cols = b.columns;
        let mut data = vec![0.; c_cols * c_rows];
        let (tx, rx): (mpsc::Sender<MatrixElement>, mpsc::Receiver<MatrixElement>) = mpsc::channel();
        let mut children = Vec::new();

        let a = Arc::new(a.clone());
        let b = Arc::new(b.clone());

        let threads = 16;
        let elements_per_thread = c_cols*c_rows / threads;
        let mut start_element = 0usize;
        for thread in 0..threads{
            let start = start_element;
            let mut end = (thread + 1) * elements_per_thread;

            if c_cols*c_rows != elements_per_thread*threads && (thread+1) == threads{
                end = (thread + 1) * elements_per_thread + 1;
            }

            let elements: Vec<usize> = (start..end).collect();
            let thread_tx = tx.clone();
            let a = Arc::clone(&a);
            let b = Arc::clone(&b);
            let child = thread::spawn(move || {
                for element in elements{
                    let row = element / c_cols;
                    let col = element % c_cols;

                    let mut sum: f64 = 0.;
                    for k in 0..a.columns {
                        sum += a.data[row][k] * b.data[k][col];
                    }
                    thread_tx.send(MatrixElement { row, column: col, value: sum }).unwrap();
                }
            });
            children.push(child);
            start_element += elements_per_thread;
        }

        drop(tx); // Close the sending side of the channel

        for element in rx {
            data[element.row * c_cols + element.column] = element.value;
        }

        for child in children {
            child.join().expect("oops! the child thread panicked");
        }

        Matrix::new(c_rows, c_cols, &data)
    }
}