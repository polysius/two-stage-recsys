//! A safe wrapper around single-precision matrix multiplication.

/// A dense row-major matrix view, optionally read as its transpose.
#[derive(Clone, Copy)]
pub struct MatrixView<'a> {
    pub data: &'a [f32],
    /// Rows and columns of the matrix as stored.
    pub stored_rows: usize,
    pub stored_cols: usize,
    pub transposed: bool,
}

impl<'a> MatrixView<'a> {
    pub fn new(data: &'a [f32], rows: usize, cols: usize) -> MatrixView<'a> {
        assert_eq!(
            data.len(),
            rows * cols,
            "matrix data does not match its shape"
        );
        MatrixView {
            data,
            stored_rows: rows,
            stored_cols: cols,
            transposed: false,
        }
    }

    pub fn t(self) -> MatrixView<'a> {
        MatrixView {
            transposed: !self.transposed,
            ..self
        }
    }

    pub fn rows(&self) -> usize {
        if self.transposed {
            self.stored_cols
        } else {
            self.stored_rows
        }
    }

    pub fn cols(&self) -> usize {
        if self.transposed {
            self.stored_rows
        } else {
            self.stored_cols
        }
    }

    /// Distance in memory between consecutive rows and consecutive columns of the view.
    fn strides(&self) -> (isize, isize) {
        if self.transposed {
            (1, self.stored_cols as isize)
        } else {
            (self.stored_cols as isize, 1)
        }
    }
}

/// `out = left * right`, or `out += left * right` when `accumulate` is set.
pub fn matmul(left: MatrixView, right: MatrixView, out: &mut [f32], accumulate: bool) {
    let (m, k, n) = (left.rows(), left.cols(), right.cols());
    assert_eq!(k, right.rows(), "inner dimensions differ");
    assert_eq!(out.len(), m * n, "output has the wrong size");
    let (row_stride_left, col_stride_left) = left.strides();
    let (row_stride_right, col_stride_right) = right.strides();
    let beta = if accumulate { 1.0 } else { 0.0 };
    // SAFETY: the asserts above make every index the strides can reach lie inside the slices.
    unsafe {
        matrixmultiply::sgemm(
            m,
            k,
            n,
            1.0,
            left.data.as_ptr(),
            row_stride_left,
            col_stride_left,
            right.data.as_ptr(),
            row_stride_right,
            col_stride_right,
            beta,
            out.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplies_plain_and_transposed_matrices() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 x 3
        let b = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3 x 2
        let mut out = [0.0; 4];
        matmul(
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&b, 3, 2),
            &mut out,
            false,
        );
        assert_eq!(out, [4.0, 5.0, 10.0, 11.0]);

        // a (2 x 3) times a-transposed (3 x 2)
        let mut gram = [0.0; 4];
        matmul(
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&a, 2, 3).t(),
            &mut gram,
            false,
        );
        assert_eq!(gram, [14.0, 32.0, 32.0, 77.0]);

        matmul(
            MatrixView::new(&a, 2, 3),
            MatrixView::new(&a, 2, 3).t(),
            &mut gram,
            true,
        );
        assert_eq!(gram, [28.0, 64.0, 64.0, 154.0]);
    }
}
