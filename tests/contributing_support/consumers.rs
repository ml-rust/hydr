use super::fixture::{Result, batch, face, measured, nonhydrological, ordered, primary};
use hydr::model::{CellState, FlowState};
use hydr::runoff::{CellRainAssignment, SupportError};
use hydr::soil::{LandModel, SoilState};
use hydr::solver::{
    BoundaryForcing, ExteriorBoundary, ExteriorState, InitialState, Solver, SolverConfig,
    SolverSetup, StageBoundary,
};

#[test]
fn observed_upstream_rain_is_suppressed_without_creating_a_hydraulic_wall() -> Result {
    let fixture = primary()?;
    let mask = fixture.domain.active().to_vec();
    let plan = fixture.plan()?;
    let packet = plan.admit_interval(fixture.interval.clone())?;
    assert_eq!(packet.forcing().rain, vec![0.0, 0.001]);
    assert_eq!(fixture.domain.active(), mask.as_slice());
    let receipt = packet
        .suppression_receipts()
        .first()
        .ok_or("missing suppression receipt")?;
    assert_eq!(packet.suppression_receipts().len(), 1);
    assert_eq!(receipt.cell, 0);
    assert_eq!(receipt.atoms, vec!["upstream"]);
    assert_eq!(receipt.observed_rain_m_per_s, 0.001);
    assert_eq!(receipt.injected_rain_m_per_s, 0.0);
    assert_eq!(receipt.selected_contribution_id, "upstream-source");
    let forcing = plan.surface_schedule(&[packet])?;
    let land = LandModel::impervious(&fixture.domain);
    let soil = SoilState::initialized(&land)?;
    let flow = FlowState::new(&fixture.domain, vec![CellState::default(); 2])?;
    let mut solver = Solver::<crate::runtime::Runtime>::new(
        crate::runtime::client(),
        SolverSetup {
            domain: fixture.domain.clone(),
            land,
            boundaries: fixture.boundaries.clone(),
            forcing,
            config: SolverConfig::default(),
        },
        InitialState {
            time: 0.0,
            flow,
            soil,
        },
    )?;
    solver.advance_to(1.0, 10)?;
    assert_eq!(solver.domain().active(), mask.as_slice());
    assert!(solver.state()?.cells()[0].h > 0.0);
    Ok(())
}

#[test]
fn active_models_receive_real_trials_in_original_definition_order() -> Result {
    let fixture = ordered()?;
    let plan = fixture.plan()?;
    assert_eq!(plan.active_definition_indices(), &[0, 1]);
    assert_eq!(plan.active_definitions()[0].id, "model-b");
    assert_eq!(plan.active_definitions()[1].id, "model-a");
    let packet = plan.admit_interval(fixture.interval)?;
    assert_eq!(packet.subcatchment_inputs()[0].rain_rate_m_per_s, 0.003);
    assert_eq!(packet.subcatchment_inputs()[1].rain_rate_m_per_s, 0.002);
    let accepted = batch(&plan, 0.0)?;
    let before = accepted.snapshot_data()?;
    let trial = plan
        .trial_subcatchments(&packet, Some(&accepted))?
        .ok_or("missing active numerical trial")?;
    let next = trial.into_next();
    assert_eq!(next.time(), 1.0);
    assert_eq!(next.states()?[0].depths_m[1], 0.003);
    assert_eq!(next.states()?[1].depths_m[1], 0.002);
    assert_eq!(accepted.snapshot_data()?, before);
    Ok(())
}

#[test]
fn whole_measured_replacement_uses_one_unchanged_boundary_and_no_model_batch() -> Result {
    let fixture = measured()?;
    let plan = fixture.plan()?;
    assert!(plan.active_definition_indices().is_empty());
    assert!(plan.active_definitions().is_empty());
    let original = fixture.interval.boundaries[0].value.clone();
    let packet = plan.admit_interval(fixture.interval)?;
    assert_eq!(packet.forcing().rain, vec![0.0; 2]);
    assert_eq!(packet.suppression_receipts().len(), 2);
    assert!(packet.subcatchment_inputs().is_empty());
    assert!(
        plan.trial_subcatchments::<crate::runtime::Runtime>(&packet, None)?
            .is_none()
    );
    let schedule = plan.surface_schedule(&[packet])?;
    assert_eq!(schedule.intervals()[0].boundaries, vec![original]);
    Ok(())
}

#[test]
fn direct_rain_evaporation_and_withdrawal_keep_the_supplied_values() -> Result {
    let mut fixture = primary()?;
    fixture.definitions.clear();
    fixture.input.contributions.clear();
    for cell in &mut fixture.input.surface_cells {
        cell.rain = CellRainAssignment::Direct;
    }
    fixture.interval.subcatchments.clear();
    fixture.interval.surface.potential_evaporation = vec![0.0001, 0.0002];
    fixture.interval.surface.withdrawal = vec![0.00001, 0.00002];
    let plan = fixture.plan()?;
    let packet = plan.admit_interval(fixture.interval.clone())?;
    assert!(packet.suppression_receipts().is_empty());
    assert_eq!(packet.forcing().rain, fixture.interval.surface.rain);
    assert_eq!(
        packet.forcing().potential_evaporation,
        fixture.interval.surface.potential_evaporation
    );
    assert_eq!(
        packet.forcing().withdrawal,
        fixture.interval.surface.withdrawal
    );
    assert!(
        plan.trial_subcatchments::<crate::runtime::Runtime>(&packet, None)?
            .is_none()
    );
    Ok(())
}

#[test]
fn nonhydrological_boundary_states_pass_through_with_complete_identity() -> Result {
    let fixture = nonhydrological()?;
    let plan = fixture.plan()?;
    for value in [
        BoundaryForcing::Stage(StageBoundary {
            face: face(),
            elevation: 1.0,
            inflow_tangential_velocity: Some(0.0),
        }),
        BoundaryForcing::Exterior(ExteriorBoundary {
            face: face(),
            state: ExteriorState::Wet {
                elevation: 1.0,
                outward_velocity: -0.1,
                tangential_velocity: 0.2,
            },
        }),
        BoundaryForcing::Transmissive(face()),
    ] {
        let mut input = fixture.interval.clone();
        input.boundaries[0].value = value.clone();
        let packet = plan.admit_interval(input)?;
        assert_eq!(packet.forcing().boundaries, vec![value]);
        plan.surface_schedule(&[packet])?;
    }
    let mut missing = fixture.input.clone();
    missing.boundaries[0].provenance.clear();
    assert!(matches!(
        hydr::runoff::SupportPlan::new(
            missing,
            &fixture.domain,
            &fixture.boundaries,
            &fixture.definitions
        ),
        Err(SupportError::Admission { .. })
    ));
    Ok(())
}
