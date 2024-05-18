use std::thread;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use ndarray::ArrayView;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD, Array2, IxDyn, Dim};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyArrayMethods, PyArray2, PyArray};
use pyo3::{pymodule, types::PyModule, PyResult, Python, Bound};

#[allow(dead_code)]
pub mod zpr_matrix{
    use ndarray::ShapeBuilder;
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

        pub fn shape(&self) -> (usize, usize){
            (self.rows, self.columns)
        }

        pub fn data(&self) -> Vec<f64>{
            let mut vec = Vec::with_capacity(self.rows*self.columns);
            for row in &self.data{
                for element in row{
                    vec.push(element.clone());
                }
            }
            vec
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

        let mut threads = 16;
        if threads > c_cols*c_rows{
            threads = c_cols*c_rows;
        }
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


#[pymodule]
fn zpr_multithread_matrix(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "multiply_f64")]
    fn multiply_f64<'py>(
        py: Python<'py>,
        x: &PyAny,
        y: &PyAny,
    ) -> PyResult<&'py PyArrayDyn<f64>> {
        let x_array = if let Ok(array) = x.extract::<PyReadonlyArrayDyn<f64>>() {
            array
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Input x must be either a numpy array of dtype float64 or int32"));
        };

        // Convert y to PyReadonlyArrayDyn<f64>
        let y_array = if let Ok(array) = y.extract::<PyReadonlyArrayDyn<f64>>() {
            array
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err("Input y must be either a numpy array of dtype float64 or int32"));
        };

        let x = x_array.as_array();
        let y = y_array.as_array();

        if x.ndim() != 2 || y.ndim() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err("Input arrays must be 2-dimensional"));
        }

        let x_shape = x.shape();
        let y_shape = y.shape();

        if x_shape[1] != y_shape[0] {
            return Err(pyo3::exceptions::PyValueError::new_err("Matrices cannot be multiplied due to incompatible dimensions"));
        }

        let x_vec: Vec<f64> = x.iter().cloned().collect();
        let y_vec: Vec<f64> = y.iter().cloned().collect();

        let x_mat = zpr_matrix::Matrix::new(x_shape[0], x_shape[1], &x_vec);
        let y_mat = zpr_matrix::Matrix::new(y_shape[0], y_shape[1], &y_vec);

        let z_mat = zpr_matrix::multi_threaded_multiply_channel(&x_mat, &y_mat);

        let shape = vec![z_mat.shape().0, z_mat.shape().1];
        let array = ArrayD::from_shape_vec(IxDyn(&shape), z_mat.data()).unwrap();
        Ok(array.into_pyarray(py))
    }
    Ok(())
}
