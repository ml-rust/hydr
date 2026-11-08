//! Checks immutable domain geometry and physical initial-state contracts.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, ModelError, SpatialReference};

fn reference() -> Result<SpatialReference, ModelError> {
    SpatialReference::new("EPSG:3375".into(), "local benchmark".into(), [10.0, 20.0])
}

fn domain() -> Result<Domain, Box<dyn std::error::Error>> {
    Ok(Domain::new(
        CartesianGrid::new(2, 1, 1.0, 1.0)?,
        vec![2.0, f64::NAN],
        vec![0.03; 2],
        vec![true, false],
        reference()?,
    )?)
}

#[test]
fn domain_normalizes_unknown_inactive_elevations() -> Result<(), Box<dyn std::error::Error>> {
    let domain = domain()?;
    assert_eq!(domain, domain.clone());
    assert_eq!(domain.bed(), &[2.0, 0.0]);
    assert_eq!(domain.manning(), &[0.03, 0.03]);
    assert_eq!(domain.active(), &[true, false]);
    assert_eq!(domain.reference().projected_crs(), "EPSG:3375");
    assert_eq!(domain.reference().vertical_datum(), "local benchmark");
    assert_eq!(domain.reference().origin(), [10.0, 20.0]);
    Ok(())
}

#[test]
fn domain_rejects_invalid_shapes_masks_beds_and_roughness() -> Result<(), Box<dyn std::error::Error>>
{
    let grid = CartesianGrid::new(2, 1, 1.0, 1.0)?;
    let cases = [
        (vec![0.0], vec![0.0; 2], vec![true; 2], "bed"),
        (vec![0.0; 2], vec![0.0], vec![true; 2], "manning"),
        (vec![0.0; 2], vec![0.0; 2], vec![true], "active"),
        (vec![0.0; 2], vec![0.0; 2], vec![false; 2], "active"),
        (vec![f64::NAN; 2], vec![0.0; 2], vec![true; 2], "bed"),
        (vec![0.0; 2], vec![-0.1; 2], vec![true; 2], "manning"),
        (
            vec![0.0; 2],
            vec![f64::INFINITY; 2],
            vec![true; 2],
            "manning",
        ),
    ];
    for (bed, manning, active, expected_field) in cases {
        let result = Domain::new(grid, bed, manning, active, reference()?);
        assert!(matches!(
            result,
            Err(ModelError::InvalidInput { field, .. }) if field == expected_field
        ));
    }
    Ok(())
}

#[test]
fn reference_rejects_empty_identifiers_and_nonfinite_coordinates() {
    for (crs, datum, origin) in [
        (" ", "datum", [0.0, 0.0]),
        ("crs", "", [0.0, 0.0]),
        ("crs", "datum", [f64::NAN, 0.0]),
        ("crs", "datum", [0.0, f64::INFINITY]),
    ] {
        assert!(SpatialReference::new(crs.into(), datum.into(), origin).is_err());
    }
}

#[test]
fn state_preserves_positive_films_and_requires_zero_inactive_state()
-> Result<(), Box<dyn std::error::Error>> {
    let domain = domain()?;
    let film = CellState {
        h: 1e-100,
        qx: 2e-100,
        qy: -3e-100,
    };
    let cells = vec![film, CellState::default()];
    let state = FlowState::new(&domain, cells.clone())?;
    assert_eq!(state.cells(), cells);
    assert_eq!(state, state.clone());
    assert!(FlowState::new(&domain, vec![film; 2]).is_err());
    assert!(FlowState::new(&domain, vec![film]).is_err());
    Ok(())
}

#[test]
fn state_rejects_invalid_depth_momentum_and_velocity() -> Result<(), Box<dyn std::error::Error>> {
    let domain = domain()?;
    for cell in [
        CellState {
            h: -1.0,
            ..CellState::default()
        },
        CellState {
            h: f64::NAN,
            ..CellState::default()
        },
        CellState {
            h: f64::INFINITY,
            ..CellState::default()
        },
        CellState {
            qx: 1.0,
            ..CellState::default()
        },
        CellState {
            qy: -1.0,
            ..CellState::default()
        },
        CellState {
            h: 1.0,
            qx: f64::INFINITY,
            qy: 0.0,
        },
        CellState {
            h: 1.0,
            qx: 0.0,
            qy: f64::NAN,
        },
        CellState {
            h: f64::MIN_POSITIVE,
            qx: f64::MAX,
            qy: 0.0,
        },
    ] {
        assert!(FlowState::new(&domain, vec![cell, CellState::default()]).is_err());
    }
    Ok(())
}

#[test]
fn state_rejects_unrepresentable_elevation_and_volume() -> Result<(), Box<dyn std::error::Error>> {
    for (spacing, bed, depths) in [
        (1.0, vec![f64::MAX; 2], vec![f64::MAX; 2]),
        (2.0, vec![0.0; 2], vec![f64::MAX; 2]),
        (1.0, vec![0.0; 2], vec![f64::MAX; 2]),
        (1e-100, vec![0.0; 2], vec![1e-200; 2]),
    ] {
        let domain = Domain::new(
            CartesianGrid::new(2, 1, spacing, spacing)?,
            bed,
            vec![0.0; 2],
            vec![true; 2],
            SpatialReference::new(
                "synthetic metric plane".into(),
                "synthetic datum".into(),
                [0.0, 0.0],
            )?,
        )?;
        let cells = depths
            .into_iter()
            .map(|h| CellState {
                h,
                ..CellState::default()
            })
            .collect();
        assert!(FlowState::new(&domain, cells).is_err());
    }
    Ok(())
}

#[test]
fn input_errors_identify_the_field_index_and_required_action() {
    let error = ModelError::InvalidInput {
        field: "h",
        index: Some(3),
        reason: "depth -1 is invalid: supply finite nonnegative metres".into(),
    };
    assert_eq!(
        error.to_string(),
        "invalid h[3]: depth -1 is invalid: supply finite nonnegative metres"
    );
}
