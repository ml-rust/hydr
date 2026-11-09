//! Checks lossless checkpoint packages and atomic rejection of inconsistent imports.

use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::solver::{
    BoundaryConditions, ForcingInterval, ForcingSchedule, SNAPSHOT_SCHEMA, SnapshotData, Solver,
    SolverConfig,
};

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn create(depth: f64) -> Result<Solver, Box<dyn std::error::Error>> {
    let domain = Domain::new(
        CartesianGrid::new(2, 1, 1.0, 2.0)?,
        vec![0.0; 2],
        vec![0.0; 2],
        vec![true; 2],
        SpatialReference::new("EPSG:3375".into(), "synthetic SI datum".into(), [0.0; 2])?,
    )?;
    let state = FlowState::new(
        &domain,
        vec![
            CellState {
                h: depth,
                qx: 0.0,
                qy: 0.0
            };
            2
        ],
    )?;
    let boundaries = BoundaryConditions::impermeable_walls();
    let forcing = ForcingSchedule::new(
        &domain,
        &boundaries,
        vec![
            ForcingInterval {
                start: 0.0,
                end: 0.4,
                rain: vec![0.001; 2],
                withdrawal: vec![0.0; 2],
                boundaries: vec![],
            },
            ForcingInterval {
                start: 0.4,
                end: 1.0,
                rain: vec![0.002; 2],
                withdrawal: vec![0.0; 2],
                boundaries: vec![],
            },
        ],
    )?;
    Ok(Solver::new(
        domain,
        state,
        boundaries,
        forcing,
        SolverConfig::default(),
        0.0,
    )?)
}

#[test]
fn package_restart_matches_trusted_continuation_exactly() -> TestResult {
    let mut original = create(1.0)?;
    original.advance_to(0.5, 100)?;
    let snapshot = original.snapshot();
    let package = snapshot.data();
    assert_eq!(package.schema, SNAPSHOT_SCHEMA);
    let mut trusted = create(0.0)?;
    let mut imported = create(0.0)?;
    trusted.restore(&snapshot)?;
    imported.restore_data(&package)?;
    assert_eq!(imported.snapshot(), snapshot);
    assert_eq!(imported.snapshot().data(), package);
    original.advance_to(1.0, 100)?;
    trusted.advance_to(1.0, 100)?;
    imported.advance_to(1.0, 100)?;
    assert_eq!(original.snapshot(), trusted.snapshot());
    assert_eq!(original.snapshot(), imported.snapshot());
    for (left, right) in original
        .state()
        .cells()
        .iter()
        .zip(imported.state().cells())
    {
        assert_eq!(left.h.to_bits(), right.h.to_bits());
        assert_eq!(left.qx.to_bits(), right.qx.to_bits());
        assert_eq!(left.qy.to_bits(), right.qy.to_bits());
    }
    assert_eq!(original.time().to_bits(), imported.time().to_bits());
    assert_eq!(
        original.ledger().storage.to_bits(),
        imported.ledger().storage.to_bits()
    );
    assert_eq!(
        original.ledger().residual.to_bits(),
        imported.ledger().residual.to_bits()
    );
    let mut endpoint = create(0.0)?;
    endpoint.restore_data(&imported.snapshot().data())?;
    assert_eq!(endpoint.snapshot(), imported.snapshot());
    Ok(())
}

#[test]
fn package_preserves_signed_zero_bits() -> TestResult {
    let original = create(0.0)?;
    let mut data = original.snapshot().data();
    data.state = FlowState::new(
        &data.domain,
        vec![
            CellState {
                h: -0.0,
                qx: -0.0,
                qy: 0.0
            };
            2
        ],
    )?;
    let mut imported = create(1.0)?;
    imported.restore_data(&data)?;
    for cell in imported.state().cells() {
        assert_eq!(cell.h.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cell.qx.to_bits(), (-0.0_f64).to_bits());
    }
    let roundtrip = imported.snapshot().data();
    for cell in roundtrip.state.cells() {
        assert_eq!(cell.h.to_bits(), (-0.0_f64).to_bits());
        assert_eq!(cell.qx.to_bits(), (-0.0_f64).to_bits());
    }
    Ok(())
}

#[test]
fn initial_package_preserves_state_and_unbounded_step_limits() -> TestResult {
    let original = create(0.0)?;
    let mut imported = create(1.0)?;
    imported.restore_data(&original.snapshot().data())?;
    assert_eq!(imported.snapshot(), original.snapshot());
    imported.advance_to(0.01, 1)?;
    let package = imported.snapshot().data();
    let step = package.last_step.as_ref().ok_or("missing accepted step")?;
    assert_eq!(step.depletion_limit, f64::INFINITY);
    let mut resumed = create(1.0)?;
    resumed.restore_data(&package)?;
    assert_eq!(resumed.snapshot().data(), package);
    Ok(())
}

#[test]
fn corrupt_packages_preserve_destination_snapshot() -> TestResult {
    let mut source = create(1.0)?;
    source.advance_to(0.5, 100)?;
    let good = source.snapshot().data();
    let mut corrupt = Vec::<SnapshotData>::new();
    let mut add = |modify: fn(&mut SnapshotData)| {
        let mut data = good.clone();
        modify(&mut data);
        corrupt.push(data);
    };
    add(|data| data.schema = "unknown".into());
    add(|data| data.contract = "unknown".into());
    add(|data| data.config.gravity = 10.0);
    add(|data| data.time = f64::NAN);
    add(|data| data.time = 2.0);
    add(|data| data.time = 0.25);
    add(|data| data.ledger.storage += 1.0);
    add(|data| data.ledger.rain = -1.0);
    add(|data| data.ledger.residual = 1.0);
    add(|data| data.ledger.water_adjustment = 1.0);
    add(|data| data.ledger.momentum_removed[0] = f64::INFINITY);
    add(|data| data.accepted_steps = 0);
    add(|data| data.last_step = None);
    let mut changed_state = good.clone();
    let mut cells = changed_state.state.cells().to_vec();
    cells[0].h += 1.0;
    changed_state.state = FlowState::new(&changed_state.domain, cells)?;
    corrupt.push(changed_state);
    let mut destination = create(0.2)?;
    destination.advance_to(0.02, 100)?;
    let before = destination.snapshot();
    for (index, data) in corrupt.iter().enumerate() {
        assert!(
            destination.restore_data(data).is_err(),
            "corruption {index}"
        );
        assert_eq!(destination.snapshot(), before, "corruption {index}");
    }
    Ok(())
}

#[test]
fn corrupt_diagnostics_preserve_destination_snapshot() -> TestResult {
    let mut source = create(1.0)?;
    source.advance_to(0.5, 100)?;
    let good = source.snapshot().data();
    let mut destination = create(0.2)?;
    let before = destination.snapshot();
    let changes: &[fn(&mut hydr::solver::StepDiagnostics)] = &[
        |step| step.start = -1.0,
        |step| step.dt *= 2.0,
        |step| step.cfl_limit = f64::NAN,
        |step| step.depletion_limit = 0.0,
        |step| step.retries = usize::MAX,
        |step| step.storage_after += 1.0,
        |step| step.rain_volume += 1.0,
        |step| step.normalized_residual = 1.0,
        |step| step.max_local_residual = 1.0,
        |step| {
            step.faces.pop();
        },
        |step| step.faces[0].left = Some(0),
        |step| step.faces[0].face.column += 1,
        |step| step.faces[0].flux[1] = f64::NAN,
        |step| step.faces[0].pressure_corrections[0] = -1.0,
        |step| step.faces[0].volume += 1.0,
    ];
    for (index, change) in changes.iter().enumerate() {
        let mut data = good.clone();
        change(data.last_step.as_mut().ok_or("missing accepted step")?);
        assert!(
            destination.restore_data(&data).is_err(),
            "corruption {index}"
        );
        assert_eq!(destination.snapshot(), before, "corruption {index}");
    }
    Ok(())
}
