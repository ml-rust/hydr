//! Complete soil continuation and source-accounting checks at owned snapshot boundaries.

#[path = "land_sources/fixture.rs"]
mod fixture;

use fixture::{Result, SourceCase};
use hydr::model::CellState;
use hydr::soil::{
    LandCell, LandModel, ModifiedGreenAmptParameters, NativeSoilState, SoilCellState, SoilState,
};
use hydr::solver::{ForcingSchedule, SnapshotData, Solver};

fn continued_case() -> Result<Solver> {
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let mut case = SourceCase::uniform(
        2,
        2,
        LandCell::ModifiedGreenAmpt(p),
        CellState {
            h: 0.001,
            qx: 0.0,
            qy: 0.0,
        },
    )?;
    case.rain.fill(1e-5);
    case.evaporation.fill(1e-6);
    case.withdrawal.fill(2e-7);
    case.config.max_step = 0.04;
    let data = case.build()?.snapshot().data();
    let mut setup = data.setup;
    let mut first = setup.forcing.intervals()[0].clone();
    first.end = 0.1;
    let mut second = first.clone();
    second.start = 0.1;
    second.end = 2.0;
    second.rain.fill(0.0);
    second.potential_evaporation.fill(2e-6);
    setup.forcing = ForcingSchedule::new(&setup.domain, &setup.boundaries, vec![first, second])?;
    Ok(Solver::new(setup, data.accepted)?)
}

#[test]
fn owned_restart_preserves_soil_and_ledgers_across_a_forcing_edge() -> Result {
    let mut continuous = continued_case()?;
    continuous.advance_to(0.1, 100)?;
    let snapshot = continuous.snapshot();
    let package = snapshot.data();
    assert_eq!(package.accepted.time, 0.1);
    assert!(package.ledger.infiltration > 0.0);
    assert!(package.ledger.evaporation > 0.0);
    let mut trusted = continued_case()?;
    let mut imported = continued_case()?;
    trusted.restore(&snapshot)?;
    imported.restore_data(&package)?;
    assert_eq!(imported.snapshot().data(), package);
    continuous.advance_to(0.2, 100)?;
    trusted.advance_to(0.2, 100)?;
    imported.advance_to(0.2, 100)?;
    assert_eq!(continuous.snapshot(), trusted.snapshot());
    assert_eq!(continuous.snapshot(), imported.snapshot());
    assert_eq!(continuous.soil_state(), imported.soil_state());
    assert_eq!(continuous.ledger(), imported.ledger());
    Ok(())
}

#[test]
fn antecedent_memory_and_signed_zeros_do_not_create_lifetime_infiltration() -> Result {
    let p = ModifiedGreenAmptParameters::from_native(0.005, 1e-5, 0.3)?;
    let native = NativeSoilState {
        imd: 0.2,
        f_ft: -0.01,
        fu_ft: 0.02,
        saturated: true,
        t_s: -20.0,
    };
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.soil = SoilState::new(&case.land, vec![SoilCellState::ModifiedGreenAmpt(native)])?;
    let original = case.build()?;
    assert_eq!(original.ledger().infiltration, 0.0);
    let mut destination =
        SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?.build()?;
    destination.restore_data(&original.snapshot().data())?;
    assert_eq!(
        destination.soil_state().cells()[0],
        SoilCellState::ModifiedGreenAmpt(native)
    );
    assert_eq!(destination.ledger().infiltration, 0.0);

    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.0)?;
    let signed = NativeSoilState {
        imd: -0.0,
        f_ft: -0.0,
        fu_ft: -0.0,
        saturated: false,
        t_s: -0.0,
    };
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.soil = SoilState::new(&case.land, vec![SoilCellState::ModifiedGreenAmpt(signed)])?;
    let original = case.build()?;
    let mut destination =
        SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?.build()?;
    destination.restore_data(&original.snapshot().data())?;
    let SoilCellState::ModifiedGreenAmpt(actual) = destination.soil_state().cells()[0] else {
        return Err("missing native soil state".into());
    };
    for value in [actual.imd, actual.f_ft, actual.fu_ft, actual.t_s] {
        assert_eq!(value.to_bits(), (-0.0_f64).to_bits());
    }
    Ok(())
}

fn adjusted_case() -> Result<Solver> {
    let p = ModifiedGreenAmptParameters::from_native(0.4, 1e-5, 0.3)?;
    let mut case = SourceCase::uniform(1, 1, LandCell::ModifiedGreenAmpt(p), CellState::default())?;
    case.rain[0] = f64::from_bits(0x3e86_d127_d053_94fe);
    case.build()
}

#[test]
fn accepted_adjustment_charge_survives_owned_restart() -> Result {
    let mut original = adjusted_case()?;
    original.advance_to(1.0, 1)?;
    let package = original.snapshot().data();
    assert!(package.ledger.water_adjustment > 0.0);
    assert!(package.ledger.water_adjustment_abs_bound >= package.ledger.water_adjustment);
    let mut resumed = adjusted_case()?;
    resumed.restore_data(&package)?;
    assert_eq!(resumed.snapshot().data(), package);
    original.advance_to(2.0, 1)?;
    resumed.advance_to(2.0, 1)?;
    assert_eq!(original.snapshot(), resumed.snapshot());
    Ok(())
}

#[test]
fn inconsistent_source_packages_preserve_the_destination() -> Result {
    let mut source = adjusted_case()?;
    source.advance_to(1.0, 1)?;
    let good = source.snapshot().data();
    let mut corrupt = Vec::new();
    let changes: &[fn(&mut SnapshotData)] = &[
        |data| data.ledger.infiltration = -1.0,
        |data| data.ledger.evaporation = -1.0,
        |data| data.ledger.water_adjustment_abs_bound = -1.0,
        |data| data.ledger.water_adjustment_abs_bound = 0.0,
        |data| data.ledger.water_adjustment_abs_bound = 1.0,
    ];
    for modify in changes {
        let mut data = good.clone();
        modify(&mut data);
        corrupt.push(data);
    }
    for quantity in 0..12 {
        let mut data = good.clone();
        let step = data.last_step.as_mut().ok_or("missing accepted step")?;
        let receipt = step.sources.first_mut().ok_or("missing source receipt")?;
        match quantity {
            0 => step.infiltration_volume += 1.0,
            1 => step.evaporation_volume += 1.0,
            2 => step.soil_water_adjustment_volume = 0.0,
            3 => step.water_adjustment_abs_charge = 0.0,
            4 => receipt.cell = usize::MAX,
            5 => receipt.ponded_depth_m = -1.0,
            6 => receipt.rain_depth_m *= 2.0,
            7 => receipt.infiltration_depth_m = 0.0,
            8 => receipt.evaporation_depth_m = 1.0,
            9 => receipt.withdrawal_depth_m = 1.0,
            10 => receipt.water_adjustment_depth_m = 0.0,
            11 => receipt.soil = None,
            _ => return Err("unknown source corruption".into()),
        }
        corrupt.push(data);
    }
    for quantity in 0..7 {
        let mut data = good.clone();
        let soil = data
            .last_step
            .as_mut()
            .ok_or("missing accepted step")?
            .sources[0]
            .soil
            .as_mut()
            .ok_or("missing soil receipt")?;
        match quantity {
            0 => soil.native_rate_ft_per_s = -1.0,
            1 => soil.native_available_depth_ft = 0.0,
            2 => soil.native_infiltration_depth_ft = 0.0,
            3 => soil.native_excess_depth_ft = 1.0,
            4 => soil.native_roundoff_bound_ft = f64::NAN,
            5 => soil.bridge_roundoff_bound_m = -1.0,
            6 => soil.total_roundoff_bound_m = 0.0,
            _ => return Err("unknown soil corruption".into()),
        }
        corrupt.push(data);
    }
    let mut destination = adjusted_case()?;
    let before = destination.snapshot();
    for (index, package) in corrupt.iter().enumerate() {
        assert!(
            destination.restore_data(package).is_err(),
            "source corruption {index}"
        );
        assert_eq!(destination.snapshot(), before, "source corruption {index}");
    }
    Ok(())
}

#[test]
fn immutable_land_and_evaporation_require_exact_definition_identity() -> Result {
    let mut source = adjusted_case()?;
    source.advance_to(1.0, 1)?;
    let good = source.snapshot().data();
    let mut corrupt = Vec::new();
    let mut changed_land = good.clone();
    changed_land.setup.land = LandModel::impervious(&changed_land.setup.domain);
    corrupt.push(changed_land);
    let mut changed_evaporation = good.clone();
    let mut intervals = changed_evaporation.setup.forcing.intervals().to_vec();
    intervals[0].potential_evaporation[0] = -0.0;
    changed_evaporation.setup.forcing = ForcingSchedule::new(
        &changed_evaporation.setup.domain,
        &changed_evaporation.setup.boundaries,
        intervals,
    )?;
    corrupt.push(changed_evaporation);
    let mut destination = adjusted_case()?;
    let before = destination.snapshot();
    for package in corrupt {
        assert!(destination.restore_data(&package).is_err());
        assert_eq!(destination.snapshot(), before);
    }
    Ok(())
}
