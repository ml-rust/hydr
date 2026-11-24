//! Capability declaration of `SurfaceSolverF32` checked against admission.

use crate::runtime::{Runtime, client};
use hydr::capability::{ComponentId, Precision, UnsupportedReason};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{SurfaceSolverF32, SurfaceSolverF32Config, surface_solver_f32_capabilities};

#[test]
fn surface_solver_f32_declaration_matches_admission() -> Result<(), Box<dyn std::error::Error>> {
    let caps = surface_solver_f32_capabilities();
    assert_eq!(caps.component, ComponentId::SurfaceSolverF32);
    assert_eq!(caps.precision, &[Precision::F32]);

    let rejected_ids: Vec<&str> = caps
        .unsupported
        .iter()
        .filter(|u| u.reason == UnsupportedReason::RejectedByAdmission)
        .map(|u| u.id)
        .collect();
    assert_eq!(
        rejected_ids,
        ["invalid-f32-config", "tampered-f32-snapshot"]
    );

    let grid = CartesianGrid::new(3, 3, 10.0, 10.0)?;
    let n = grid.cell_count();
    let domain = Domain::new(
        grid,
        vec![0.0; n],
        vec![0.03; n],
        vec![true; n],
        SpatialReference::new("metric".into(), "NAVD88".into(), [0.0, 0.0])?,
    )?;
    let flow = FlowState::new(
        &domain,
        vec![
            CellState {
                h: 0.5,
                qx: 0.0,
                qy: 0.0
            };
            n
        ],
    )?;

    // 1. invalid-f32-config rejects
    let bad_cfg = SurfaceSolverF32Config {
        cfl: 0.9,
        ..SurfaceSolverF32Config::default()
    };
    assert!(
        SurfaceSolverF32::<Runtime>::new(client(), domain.clone(), flow.clone(), bad_cfg).is_err()
    );

    // 2. tampered-f32-snapshot rejects
    let mut solver = SurfaceSolverF32::<Runtime>::new(
        client(),
        domain,
        flow,
        SurfaceSolverF32Config::default(),
    )?;
    let mut forged = solver.snapshot_data()?;
    forged.schema = "forged-f32-schema".into();
    assert!(solver.restore_data(&forged).is_err());

    Ok(())
}
