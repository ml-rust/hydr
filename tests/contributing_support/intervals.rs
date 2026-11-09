use super::fixture::{Result, batch, face, measured, model_input, nonhydrological, primary};
use hydr::runoff::{SupportError, SupportErrorKind};
use hydr::solver::{
    Axis, BoundaryFace, BoundaryForcing, DischargeBoundary, FluxPort, StageBoundary,
};

fn category<T>(result: std::result::Result<T, SupportError>, expected: SupportErrorKind) -> Result {
    let error = match result {
        Ok(_) => return Err("invalid interval requires rejection".into()),
        Err(error) => error,
    };
    assert_eq!(error.kind(), Some(expected), "unexpected error: {error}");
    Ok(())
}

#[test]
fn active_inputs_require_exact_identity_complete_presence_and_no_duplicates() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    for mode in 0..5 {
        let mut input = fixture.interval.clone();
        match mode {
            0 => input.subcatchments.clear(),
            1 => input.subcatchments.push(input.subcatchments[0].clone()),
            2 => input.subcatchments.push(model_input("unknown", 0.0)),
            3 => input.subcatchments[0].input_identity = "foreign-input".into(),
            4 => input.surface.input_identity = "foreign-surface".into(),
            _ => return Err("unknown input fixture".into()),
        }
        category(
            plan.admit_interval(input),
            if mode >= 3 {
                SupportErrorKind::Identity
            } else {
                SupportErrorKind::Coverage
            },
        )?;
    }
    let mut zero = fixture.interval.clone();
    zero.subcatchments[0].input.rain_rate_m_per_s = 0.0;
    let packet = plan.admit_interval(zero)?;
    assert_eq!(packet.subcatchment_inputs()[0].rain_rate_m_per_s, 0.0);
    Ok(())
}

#[test]
fn exact_spans_precede_unpaired_fixed_volume_provenance_rejection() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    let accepted = batch(&plan, 0.0)?;
    let before = accepted.snapshot_data()?;
    let mut input = fixture.interval.clone();
    input.subcatchments[0].input.external_runon_volumes_m3[1] = 0.1;
    input.subcatchments[0].end = 2.0;
    category(
        plan.admit_interval(input.clone()),
        SupportErrorKind::Coverage,
    )?;
    input.subcatchments[0].end = 1.0;
    let error = match plan.admit_interval(input) {
        Ok(_) => return Err("unpaired volume requires rejection".into()),
        Err(error) => error,
    };
    assert_eq!(error.kind(), Some(SupportErrorKind::Composition));
    assert!(error.to_string().contains("donor provenance"));
    assert_eq!(accepted.snapshot_data()?, before);
    for mode in 0..3 {
        let mut input = fixture.interval.clone();
        match mode {
            0 => input.surface.end = 2.0,
            1 => input.subcatchments[0].start = -1.0,
            2 => input.end = 0.5,
            _ => return Err("unknown span fixture".into()),
        }
        category(plan.admit_interval(input), SupportErrorKind::Coverage)?;
    }
    Ok(())
}

#[test]
fn suppressed_model_requests_reject_even_when_every_value_is_zero() -> Result {
    let fixture = measured()?;
    let plan = fixture.plan()?;
    let mut input = fixture.interval;
    input.subcatchments.push(model_input("source-a", 0.0));
    category(plan.admit_interval(input), SupportErrorKind::Coverage)
}

#[test]
fn boundary_packets_require_exact_count_identity_destination_and_span() -> Result {
    let fixture = measured()?;
    let plan = fixture.plan()?;
    for mode in 0..6 {
        let mut input = fixture.interval.clone();
        let expected = match mode {
            0 => {
                input.boundaries.clear();
                SupportErrorKind::Coverage
            }
            1 => {
                input.boundaries.push(input.boundaries[0].clone());
                SupportErrorKind::Coverage
            }
            2 => {
                input.boundaries[0].boundary_id = "foreign-boundary".into();
                SupportErrorKind::Identity
            }
            3 => {
                input.boundaries[0].input_identity = "foreign-input".into();
                SupportErrorKind::Identity
            }
            4 => {
                input.boundaries[0].end = 2.0;
                SupportErrorKind::Coverage
            }
            5 => {
                input.boundaries[0].value = BoundaryForcing::Discharge(DischargeBoundary {
                    face: BoundaryFace {
                        axis: Axis::X,
                        column: 2,
                        row: 0,
                    },
                    outward_rate: -0.003,
                    inflow_tangential_velocity: Some(0.0),
                });
                SupportErrorKind::Destination
            }
            _ => return Err("unknown boundary input fixture".into()),
        };
        category(plan.admit_interval(input), expected)?;
    }
    let mut stage = fixture.interval;
    stage.boundaries[0].value = BoundaryForcing::Stage(StageBoundary {
        face: face(),
        elevation: 1.0,
        inflow_tangential_velocity: Some(0.0),
    });
    category(plan.admit_interval(stage), SupportErrorKind::Composition)
}

#[test]
fn unassociated_incoming_rates_require_named_sources_without_implicit_composition() -> Result {
    let fixture = nonhydrological()?;
    let plan = fixture.plan()?;
    for rate in [-0.001, 0.0, 0.001] {
        for flux in [true, false] {
            let mut input = fixture.interval.clone();
            input.boundaries[0].value = if flux {
                BoundaryForcing::Flux(FluxPort {
                    face: face(),
                    water_rate: rate,
                    momentum_rate: [-4.905 - rate * rate, 0.0],
                    signal_speed: 4.0,
                    provenance: "One-metre depth and width with complete outward water and pressure momentum flux".into(),
                })
            } else {
                BoundaryForcing::Discharge(DischargeBoundary {
                    face: face(),
                    outward_rate: rate,
                    inflow_tangential_velocity: Some(0.0),
                })
            };
            let expected = input.boundaries[0].value.clone();
            if rate < 0.0 {
                category(plan.admit_interval(input), SupportErrorKind::Composition)?;
            } else {
                let packet = plan.admit_interval(input)?;
                assert_eq!(packet.forcing().boundaries, vec![expected]);
            }
        }
    }
    Ok(())
}

#[test]
fn foreign_packets_and_gapped_schedules_never_rewrite_admitted_data() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    let foreign = fixture.plan()?.admit_interval(fixture.interval.clone())?;
    category(
        plan.surface_schedule(&[foreign]),
        SupportErrorKind::Identity,
    )?;
    let first = plan.admit_interval(fixture.interval.clone())?;
    let mut input = fixture.interval;
    input.start = 2.0;
    input.end = 3.0;
    input.surface.start = 2.0;
    input.surface.end = 3.0;
    input.subcatchments[0].start = 2.0;
    input.subcatchments[0].end = 3.0;
    let second = plan.admit_interval(input)?;
    assert!(matches!(
        plan.surface_schedule(&[first.clone(), second]),
        Err(SupportError::Surface(_))
    ));
    assert_eq!(first.forcing().rain, vec![0.0, 0.001]);
    Ok(())
}

#[test]
fn invalid_or_inactive_surface_observations_remain_numerical_input_errors() -> Result {
    let fixture = primary()?;
    let plan = fixture.plan()?;
    let mut invalid = fixture.interval.clone();
    invalid.surface.rain[0] = f64::NAN;
    assert!(matches!(
        plan.admit_interval(invalid),
        Err(SupportError::Surface(_))
    ));
    let mut inactive = primary()?;
    inactive.domain = super::fixture::domain(2, 1.0, vec![true, false])?;
    inactive.input.atoms.retain(|atom| atom.id == "upstream");
    inactive.input.surface_cells.retain(|cell| cell.cell == 0);
    let plan = inactive.plan()?;
    let mut valid = inactive.interval.clone();
    valid.surface.rain[1] = 0.0;
    plan.admit_interval(valid)?;
    assert!(matches!(
        plan.admit_interval(inactive.interval),
        Err(SupportError::Surface(_))
    ));
    Ok(())
}
