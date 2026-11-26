//! Markdown tables in the same pipe format the Python scripts wrote with `to_markdown`.

use std::path::Path;

use anyhow::Result;

pub enum Cell {
    Text(String),
    Number(f64),
}

impl From<&str> for Cell {
    fn from(text: &str) -> Cell {
        Cell::Text(text.to_string())
    }
}

impl From<String> for Cell {
    fn from(text: String) -> Cell {
        Cell::Text(text)
    }
}

impl From<f64> for Cell {
    fn from(number: f64) -> Cell {
        Cell::Number(number)
    }
}

/// Rounds to 4 decimals and drops trailing zeros, so 0.0189 stays 0.0189 and 69.1 stays 69.1.
fn format_number(value: f64) -> String {
    if value.is_nan() {
        return "nan".to_string();
    }
    let text = format!("{value:.4}");
    let trimmed = text.trim_end_matches('0').trim_end_matches('.');
    if trimmed.is_empty() || trimmed == "-0" {
        "0".to_string()
    } else {
        trimmed.to_string()
    }
}

/// Formats like Python's `{value:.{significant}g}`: that many significant digits, switching to
/// exponent form for very small or very large values, with trailing zeros removed.
pub fn format_general(value: f64, significant: usize) -> String {
    if value == 0.0 || !value.is_finite() {
        return if value == 0.0 {
            "0".to_string()
        } else {
            format!("{value}")
        };
    }
    let exponent = value.abs().log10().floor() as i32;
    if exponent < -4 || exponent >= significant as i32 {
        let text = format!("{value:.digits$e}", digits = significant - 1);
        let (mantissa, exp) = text.split_once('e').expect("exponent form");
        let mantissa = if mantissa.contains('.') {
            mantissa.trim_end_matches('0').trim_end_matches('.')
        } else {
            mantissa
        };
        let exp: i32 = exp.parse().expect("integer exponent");
        return format!(
            "{mantissa}e{}{:02}",
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        );
    }
    let decimals = (significant as i32 - 1 - exponent).max(0) as usize;
    let text = format!("{value:.decimals$}");
    if text.contains('.') {
        text.trim_end_matches('0').trim_end_matches('.').to_string()
    } else {
        text
    }
}

/// A pipe table. Text columns align left and numeric columns align right.
pub fn markdown_table(headers: &[&str], rows: &[Vec<Cell>]) -> String {
    let rendered: Vec<Vec<String>> = rows
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| match cell {
                    Cell::Text(text) => text.clone(),
                    Cell::Number(number) => format_number(*number),
                })
                .collect()
        })
        .collect();
    let widths: Vec<usize> = (0..headers.len())
        .map(|col| {
            rendered
                .iter()
                .map(|row| row[col].len())
                .chain([headers[col].len(), 3])
                .max()
                .unwrap()
        })
        .collect();
    let is_numeric = |col: usize| {
        rows.first()
            .is_some_and(|row| matches!(row[col], Cell::Number(_)))
    };

    let pad = |text: &str, col: usize| {
        if is_numeric(col) {
            format!("{text:>width$}", width = widths[col])
        } else {
            format!("{text:<width$}", width = widths[col])
        }
    };
    let line = |cells: Vec<String>| format!("| {} |", cells.join(" | "));

    let mut lines = vec![line(
        (0..headers.len())
            .map(|col| pad(headers[col], col))
            .collect(),
    )];
    lines.push(line(
        (0..headers.len())
            .map(|col| {
                let dashes = "-".repeat(widths[col] - 1);
                if is_numeric(col) {
                    format!("{dashes}:")
                } else {
                    format!(":{dashes}")
                }
            })
            .collect(),
    ));
    for row in &rendered {
        lines.push(line(
            (0..headers.len()).map(|col| pad(&row[col], col)).collect(),
        ));
    }
    lines.join("\n")
}

pub fn write_report(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, text)?;
    println!("wrote {}", path.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_are_rounded_and_trimmed() {
        assert_eq!(format_number(0.018912), "0.0189");
        assert_eq!(format_number(69.1), "69.1");
        assert_eq!(format_number(3.0), "3");
        assert_eq!(format_number(0.00001), "0");
    }

    #[test]
    fn general_format_matches_python() {
        assert_eq!(format_general(0.0123456, 2), "0.012");
        assert_eq!(format_general(0.5, 2), "0.5");
        assert_eq!(format_general(0.000012345, 2), "1.2e-05");
        assert_eq!(format_general(1.0e-20, 2), "1e-20");
        assert_eq!(format_general(123456.0, 2), "1.2e+05");
        assert_eq!(format_general(0.0, 2), "0");
    }

    #[test]
    fn table_aligns_text_left_and_numbers_right() {
        let table = markdown_table(&["model", "recall"], &[vec!["als".into(), 0.5.into()]]);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines[1], "| :---- | -----: |");
        assert_eq!(lines[2], "| als   |    0.5 |");
    }
}
