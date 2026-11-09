//! Atomic land-source priorities, momentum, soil memory, and numerical adjustment budgets.

#[path = "land_sources/fixture.rs"]
mod fixture;

use fixture::{Result, SourceCase, domain};
use hydr::grid::CartesianGrid;
use hydr::model::{CellState, Domain, FlowState, SpatialReference};
use hydr::soil::{
    LandCell, LandModel, ModifiedGreenAmptBatch, ModifiedGreenAmptParameters, NativeSoilState,
    SoilCellState, SoilInput, SoilState,
};
use hydr::solver::{AdvanceStatus, CellResult};

fn native_state(imd: f64, f_ft: f64, saturated: bool) -> NativeSoilState {
    NativeSoilState {
        imd,
        f_ft,
        fu_ft: 0.0,
        saturated,
        t_s: 0.0,
    }
}

fn close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e}, expected={expected:.17e}, tolerance={tolerance:.17e}"
    );
}

#[test]
fn land_and_antecedent_support_require_matching_cell_variants() -> Result {
    let domain = domain(2, 1, vec![true, false])?;
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    assert!(LandModel::new(&domain, vec![LandCell::Impervious]).is_err());
    assert!(
        LandModel::new(
            &domain,
            vec![LandCell::Impervious, LandCell::ModifiedGreenAmpt(p)],
        )
        .is_err()
    );
    let land = LandModel::new(
        &domain,
        vec![LandCell::ModifiedGreenAmpt(p), LandCell::Impervious],
    )?;
    assert!(SoilState::new(&land, vec![SoilCellState::Impervious; 2]).is_err());
    assert!(
        SoilState::new(
            &land,
            vec![SoilCellState::ModifiedGreenAmpt(native_state(0.3, 0.0, false)); 2],
        )
        .is_err()
    );
    let soil = SoilState::initialized(&land)?;
    assert_eq!(soil.cells()[1], SoilCellState::Impervious);
    Ok(())
}

#[test]
fn invalid_native_antecedent_reports_its_global_cell() -> Result {
    let domain = domain(5, 1, vec![true; 5])?;
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let land = LandModel::new(
        &domain,
        vec![
            LandCell::Impervious,
            LandCell::ModifiedGreenAmpt(p),
            LandCell::Impervious,
            LandCell::ModifiedGreenAmpt(p),
            LandCell::Impervious,
        ],
    )?;
    let initialized = SoilState::initialized(&land)?;
    let invalid = NativeSoilState {
        imd: p.recovered_deficit_ceiling().next_up(),
        ..native_state(0.3, 0.0, false)
    };
    let mut cells = initialized.cells().to_vec();
    cells[3] = SoilCellState::ModifiedGreenAmpt(invalid);
    let error = SoilState::new(&land, cells)
        .err()
        .ok_or("invalid native deficit requires rejection")?;
    assert_eq!(error.cell(), Some(3));
    Ok(())
}

#[test]
fn compact_pervious_lanes_return_to_their_global_cells() -> Result {
    let domain = domain(5, 1, vec![true, false, true, false, true])?;
    let p = ModifiedGreenAmptParameters::from_native(0.0, 1.0 / 1024.0, 0.0)?;
    let land = LandModel::new(
        &domain,
        vec![
            LandCell::ModifiedGreenAmpt(p),
            LandCell::Impervious,
            LandCell::Impervious,
            LandCell::Impervious,
            LandCell::ModifiedGreenAmpt(p),
        ],
    )?;
    let flow = FlowState::new(&domain, vec![CellState::default(); 5])?;
    let mut case = SourceCase::new(domain, land, flow)?;
    let rain = [
        0.5 / 1024.0 * 0.3048,
        0.0,
        0.0004,
        0.0,
        2.0 / 1024.0 * 0.3048,
    ];
    case.rain = rain.to_vec();
    let raw = ModifiedGreenAmptBatch::initialized(&[p, p])?.trial(
        1.0,
        &[
            SoilInput {
                incoming_rate_m_per_s: rain[0],
                ponded_depth_m: 0.0,
            },
            SoilInput {
                incoming_rate_m_per_s: rain[4],
                ponded_depth_m: 0.0,
            },
        ],
    )?;
    let outputs = raw.outputs()?;
    let expected = raw.into_next().native_states()?;
    let mut solver = case.build()?;
    solver.advance_to(1.0, 1)?;
    assert_eq!(
        solver.soil_state().cells()[0],
        SoilCellState::ModifiedGreenAmpt(expected[0])
    );
    assert_eq!(
        solver.soil_state().cells()[4],
        SoilCellState::ModifiedGreenAmpt(expected[1])
    );
    assert_eq!(solver.soil_state().cells()[2], SoilCellState::Impervious);
    assert_eq!(solver.cell(1), CellResult::Unsupported);
    assert_eq!(solver.cell(3), CellResult::Unsupported);
    close(solver.state().cells()[2].h, rain[2], 1e-18);
    close(
        solver.ledger().infiltration,
        outputs[0].depth_m + outputs[1].depth_m,
        1e-18,
    );
    Ok(())
}

#[test]
fn soil_uses_original_ponding_and_rain_once_before_evaporation() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.0, 1.0 / 512.0, 0.0)?;
    let h = 0.3048 / 1024.0;
    let rain = 0.3048 / 2048.0;
    let mut case = SourceCase::uniform(
        1,
        1,
        LandCell::ModifiedGreenAmpt(p),
        CellState {
            h,
            qx: 0.0,
            qy: 0.0,
        },
    )?;
    case.rain[0] = rain;
    case.evaporation[0] = 1.0;
    let raw = ModifiedGreenAmptBatch::initialized(&[p])?.trial(
        1.0,
        &[SoilInput {
            incoming_rate_m_per_s: rain,
            ponded_depth_m: h,
        }],
    )?;
    let output = raw.outputs()?[0];
    let expected_soil = raw.into_next().native_states()?[0];
    let mut solver = case.build()?;
    solver.advance_to(1.0, 1)?;
    assert_eq!(
        solver.soil_state().cells()[0],
        SoilCellState::ModifiedGreenAmpt(expected_soil)
    );
    assert_eq!(
        solver.ledger().infiltration.to_bits(),
        output.depth_m.to_bits()
    );
    close(solver.ledger().infiltration, h + rain, 1e-18);
    assert_eq!(solver.ledger().evaporation, 0.0);
    assert_eq!(solver.state().cells()[0], CellState::default());
    Ok(())
}

#[test]
fn impervious_evaporation_caps_at_available_water() -> Result {
    let mut case = SourceCase::uniform(
        1,
        1,
        LandCell::Impervious,
        CellState {
            h: 0.05,
            qx: 0.01,
            qy: -0.02,
        },
    )?;
    case.evaporation[0] = 1.0;
    let mut solver = case.build()?;
    solver.advance_to(1.0, 1)?;
    assert_eq!(solver.state().cells()[0], CellState::default());
    assert_eq!(solver.ledger().infiltration, 0.0);
    assert_eq!(solver.ledger().evaporation, 0.05);
    assert_eq!(solver.ledger().water_adjustment_abs_bound, 0.0);
    Ok(())
}

#[test]
fn sequential_land_sinks_remove_donor_momentum() -> Result {
    let p = ModifiedGreenAmptParameters::from_si(0.0, 0.02, 0.0)?;
    let cell = CellState {
        h: 1.0,
        qx: 0.2,
        qy: -0.1,
    };
    let mut case = SourceCase::uniform(5, 5, LandCell::ModifiedGreenAmpt(p), cell)?;
    case.rain.fill(0.1);
    case.evaporation.fill(0.03);
    case.withdrawal.fill(0.04);
    let dt = 0.01;
    let raw = ModifiedGreenAmptBatch::initialized(&[p])?.trial(
        dt,
        &[SoilInput {
            incoming_rate_m_per_s: 0.1,
            ponded_depth_m: cell.h,
        }],
    )?;
    let infiltration = raw.outputs()?[0].depth_m;
    let available = cell.h + 0.1 * dt;
    let expected_h = available - infiltration - 0.03 * dt - 0.04 * dt;
    let mut solver = case.build()?;
    solver.advance_to(dt, 1)?;
    let center = solver.state().cells()[12];
    close(center.h, expected_h, 2e-15);
    close(center.qx, cell.qx * expected_h / available, 2e-15);
    close(center.qy, cell.qy * expected_h / available, 2e-15);
    assert_eq!(
        solver
            .last_step()
            .ok_or("missing accepted step")?
            .momentum_removed,
        [0.0; 2]
    );
    Ok(())
}

#[test]
fn unmet_withdrawal_rejects_surface_soil_and_ledgers_together() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.rain[0] = 1e-6;
    case.withdrawal[0] = 1e-5;
    case.config.max_retries = 0;
    let mut solver = case.build()?;
    let before = solver.snapshot();
    assert!(solver.advance_to(1.0, 1).is_err());
    assert_eq!(solver.snapshot(), before);
    Ok(())
}

#[test]
fn potential_evaporation_requires_explicit_nonnegative_cell_support() -> Result {
    for rates in [vec![], vec![-1.0], vec![f64::NAN], vec![f64::INFINITY]] {
        let mut case = SourceCase::uniform(1, 1, LandCell::Impervious, CellState::default())?;
        case.evaporation = rates;
        assert!(case.build().is_err());
    }
    let domain = domain(2, 1, vec![true, false])?;
    let land = LandModel::impervious(&domain);
    let flow = FlowState::new(&domain, vec![CellState::default(); 2])?;
    let mut case = SourceCase::new(domain, land, flow)?;
    case.evaporation[1] = 1.0;
    assert!(case.build().is_err());
    Ok(())
}

#[test]
fn positive_source_volume_underflow_rejects_without_losing_water() -> Result {
    let domain = Domain::new(
        CartesianGrid::new(1, 1, 1e-150, 1e-150)?,
        vec![0.0],
        vec![0.0],
        vec![true],
        SpatialReference::new(
            "synthetic metric plane".into(),
            "synthetic land-source datum".into(),
            [0.0; 2],
        )?,
    )?;
    let land = LandModel::impervious(&domain);
    let flow = FlowState::new(&domain, vec![CellState::default()])?;
    let mut case = SourceCase::new(domain, land, flow)?;
    case.rain[0] = 1e-100;
    case.config.max_retries = 0;
    let mut solver = case.build()?;
    let before = solver.snapshot();
    assert!(solver.advance_to(1.0, 1).is_err());
    assert_eq!(solver.snapshot(), before);
    Ok(())
}

#[test]
fn hydraulic_shortening_restarts_soil_from_the_accepted_state() -> Result {
    let create = |max_retries| -> Result<_> {
        let p = ModifiedGreenAmptParameters::from_si(0.0, 0.1, 0.0)?;
        let mut case =
            SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
        case.rain[0] = 1.0;
        case.config.max_retries = max_retries;
        case.build()
    };
    let mut shortened = create(64)?;
    assert_eq!(
        shortened.advance_to(1.0, 1)?.status,
        AdvanceStatus::StepLimit
    );
    let step = shortened.last_step().ok_or("missing accepted step")?;
    assert!(step.retries > 0);
    assert!(shortened.time() > 0.0 && shortened.time() < 1.0);
    let mut direct = create(64)?;
    direct.advance_to(shortened.time(), 1)?;
    assert_eq!(shortened.state(), direct.state());
    assert_eq!(shortened.soil_state(), direct.soil_state());
    assert_eq!(shortened.ledger(), direct.ledger());
    let mut rejected = create(0)?;
    let before = rejected.snapshot();
    assert!(rejected.advance_to(1.0, 1).is_err());
    assert_eq!(rejected.snapshot(), before);
    Ok(())
}

fn crossing(suction: f64, memory: f64, dt: f64, budget: f64) -> Result<SourceCase> {
    let p = ModifiedGreenAmptParameters::from_native(suction, 1.0 / 1024.0, 1.0)?;
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.soil = SoilState::new(
        &case.land,
        vec![SoilCellState::ModifiedGreenAmpt(native_state(
            1.0, memory, false,
        ))],
    )?;
    case.rain[0] = 1.5 / 1024.0 * 0.3048;
    case.end = dt * 2.0;
    case.config.max_step = dt;
    case.config.max_retries = 0;
    case.config.max_cumulative_water_adjustment_m3 = budget;
    Ok(case)
}

#[test]
fn proved_native_crossing_excess_has_explicit_water_adjustment() -> Result {
    let dt = 1.02e-12;
    let mut solver = crossing(0.125, 0.25, dt, 1e-9)?.build()?;
    solver.advance_to(dt, 1)?;
    assert_eq!(solver.state().cells()[0], CellState::default());
    assert!(solver.ledger().infiltration > solver.ledger().rain);
    assert!(solver.ledger().water_adjustment > 0.0);
    assert!(solver.ledger().water_adjustment_abs_bound >= solver.ledger().water_adjustment);
    assert!(solver.ledger().water_adjustment_abs_bound <= 1e-9);
    close(
        solver.ledger().infiltration,
        solver.ledger().rain + solver.ledger().water_adjustment,
        1e-30,
    );
    let step = solver.last_step().ok_or("missing accepted step")?;
    let receipt = step.sources[0]
        .soil
        .as_ref()
        .ok_or("missing soil receipt")?;
    assert!(receipt.native_excess_depth_ft > 0.0);
    assert!(receipt.native_roundoff_bound_ft >= receipt.native_excess_depth_ft);
    assert!(receipt.total_roundoff_bound_m >= step.sources[0].water_adjustment_depth_m);
    let SoilCellState::ModifiedGreenAmpt(state) = solver.soil_state().cells()[0] else {
        return Err("missing native soil state".into());
    };
    assert_eq!(state.f_ft.to_bits(), 0.2500000000000015_f64.to_bits());
    assert!(state.saturated);
    Ok(())
}

#[test]
fn adjustment_budget_rejects_small_and_material_native_excess_atomically() -> Result {
    for mut solver in [
        crossing(0.125, 0.25, 1.02e-12, 0.0)?.build()?,
        crossing(2.0_f64.powi(52), 2.0_f64.powi(53), 1126.4, 1e-9)?.build()?,
    ] {
        let before = solver.snapshot();
        let dt = solver.config().max_step;
        assert!(solver.advance_to(dt, 1).is_err());
        assert_eq!(solver.snapshot(), before);
    }
    Ok(())
}

#[test]
fn repeated_si_bridge_adjustments_consume_one_cumulative_budget() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.rain[0] = f64::from_bits(0x3e86_d127_d053_94fe);
    let delta = 2.6469779601696886e-23;
    case.config.max_cumulative_water_adjustment_m3 = 1.5 * delta;
    case.config.max_retries = 0;
    let mut solver = case.build()?;
    solver.advance_to(1.0, 1)?;
    assert_eq!(solver.ledger().water_adjustment.to_bits(), delta.to_bits());
    assert!(solver.ledger().water_adjustment_abs_bound >= delta);
    assert!(solver.ledger().water_adjustment_abs_bound <= 1.5 * delta);
    let before = solver.snapshot();
    assert!(solver.advance_to(2.0, 1).is_err());
    assert_eq!(solver.snapshot(), before);
    Ok(())
}
