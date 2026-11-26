//! A binary sparse matrix (compressed rows) used for "who interacted with what".

use serde::{Deserialize, Serialize};

use crate::table::Table;

/// Row `r` holds the sorted, distinct column ids `indices[indptr[r]..indptr[r + 1]]`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BinaryCsr {
    pub n_cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
}

impl BinaryCsr {
    /// One entry per distinct (row, col) pair. Duplicate pairs collapse into one entry.
    pub fn from_pairs(n_rows: usize, n_cols: usize, rows: &[u32], cols: &[u32]) -> BinaryCsr {
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); n_rows];
        for (&row, &col) in rows.iter().zip(cols) {
            lists[row as usize].push(col);
        }
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::with_capacity(rows.len());
        indptr.push(0);
        for mut list in lists {
            list.sort_unstable();
            list.dedup();
            indices.extend_from_slice(&list);
            indptr.push(indices.len());
        }
        BinaryCsr {
            n_cols,
            indptr,
            indices,
        }
    }

    /// The user-item matrix: 1 when the user interacted with the item.
    pub fn user_item(table: &Table, n_users: usize, n_items: usize) -> BinaryCsr {
        BinaryCsr::from_pairs(n_users, n_items, &table.user, &table.item)
    }

    pub fn n_rows(&self) -> usize {
        self.indptr.len().saturating_sub(1)
    }

    pub fn row(&self, row: usize) -> &[u32] {
        &self.indices[self.indptr[row]..self.indptr[row + 1]]
    }

    pub fn transpose(&self) -> BinaryCsr {
        let mut rows = Vec::with_capacity(self.indices.len());
        for row in 0..self.n_rows() {
            rows.extend(std::iter::repeat_n(row as u32, self.row(row).len()));
        }
        BinaryCsr::from_pairs(self.n_cols, self.n_rows(), &self.indices, &rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn duplicates_collapse_and_rows_are_sorted() {
        let matrix = BinaryCsr::from_pairs(3, 5, &[0, 0, 0, 2], &[4, 1, 4, 3]);
        assert_eq!(matrix.row(0), &[1, 4]);
        assert!(matrix.row(1).is_empty());
        assert_eq!(matrix.row(2), &[3]);
    }

    #[test]
    fn transpose_swaps_rows_and_columns() {
        let matrix = BinaryCsr::from_pairs(2, 3, &[0, 0, 1], &[0, 2, 2]);
        let transposed = matrix.transpose();
        assert_eq!(transposed.n_rows(), 3);
        assert_eq!(transposed.row(2), &[0, 1]);
        assert_eq!(transposed.row(1), &[] as &[u32]);
    }
}
