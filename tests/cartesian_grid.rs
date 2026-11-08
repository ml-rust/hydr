//! Checks public grid geometry and rejected inputs.

use hydr::{error::GridError, grid::CartesianGrid};

#[test]
fn rectangular_geometry_uses_si_units() -> Result<(), GridError> {
    let grid = CartesianGrid::new(12, 8, 2.0, 3.0)?;
    assert_eq!(grid.dimensions(), (12, 8));
    assert_eq!(grid.spacing(), (2.0, 3.0));
    assert_eq!(grid.cell_count(), 96);
    assert_eq!(grid.cell_area(), 6.0);
    assert_eq!(grid.extent(), (24.0, 24.0));
    Ok(())
}

#[test]
fn empty_dimensions_are_rejected() {
    for (nx, ny) in [(0, 1), (1, 0), (0, 0)] {
        assert_eq!(
            CartesianGrid::new(nx, ny, 1.0, 1.0),
            Err(GridError::EmptyDimensions { nx, ny })
        );
    }
}

#[test]
fn cell_count_overflow_is_rejected() {
    assert!(matches!(
        CartesianGrid::new(usize::MAX, 2, 1.0, 1.0),
        Err(GridError::CellCountOverflow { .. })
    ));
}

#[test]
fn invalid_spacing_is_rejected_on_both_axes() {
    for spacing in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for (dx, dy) in [(spacing, 1.0), (1.0, spacing)] {
            assert!(matches!(
                CartesianGrid::new(2, 2, dx, dy),
                Err(GridError::InvalidSpacing { .. })
            ));
        }
    }
}

#[test]
fn unrepresentable_area_and_extents_are_rejected() {
    for (nx, ny, dx, dy) in [
        (1, 1, f64::MAX, 2.0),
        (1, 1, f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        (2, 1, f64::MAX, 0.5),
        (1, 2, 0.5, f64::MAX),
    ] {
        assert!(matches!(
            CartesianGrid::new(nx, ny, dx, dy),
            Err(GridError::InvalidSpacing { .. })
        ));
    }
}
