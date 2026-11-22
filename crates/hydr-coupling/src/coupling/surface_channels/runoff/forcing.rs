//! Admission of the runoff records and pool rain of one coordinator forcing interval.

use super::admission::{RunoffTarget, plan_error};
use crate::coupling::surface_channels::{
    SurfaceChannelDefinition, SurfaceChannelError, SurfaceChannelForcingInterval,
};
use crate::runoff::AdmittedRates;
use crate::solver::ForcingSchedule;

/// Admits the runoff rate records of `forcing` over its exact span, or `None` without runoff.
///
/// Rejects runoff records without a runoff setup, every plan rate rejection, and nonzero rain on
/// a pool whose rain support a selected source owns (`suppressed-rain`).
pub(in super::super) fn admit_forcing(
    definition: &SurfaceChannelDefinition,
    forcing: &SurfaceChannelForcingInterval,
) -> Result<Option<AdmittedRates>, SurfaceChannelError> {
    let Some(runoff) = &definition.runoff else {
        if let Some(record) = forcing.runoff.first() {
            return Err(SurfaceChannelError::input(format!(
                "runoff-without-setup: forcing runoff record {:?} without setup.runoff: supply no runoff records",
                record.contribution_id
            )));
        }
        return Ok(None);
    };
    if let Some(sources) = &forcing.pool_sources {
        for (pool, source) in definition.setup.pools.iter().zip(sources) {
            if runoff.plan.pool_rain_suppressed(&pool.id) && source.rain_rate_m_per_s != 0.0 {
                return Err(SurfaceChannelError::input(format!(
                    "suppressed-rain: pool={:?} rain_rate_m_per_s={} in [{},{}]: supply zero rain on a pool whose rain support a selected source owns",
                    pool.id, source.rain_rate_m_per_s, forcing.start_s, forcing.end_s
                )));
            }
        }
    }
    runoff
        .plan
        .admit_rates(forcing.start_s, forcing.end_s, &forcing.runoff)
        .map(Some)
        .map_err(|error| plan_error("runoff-forcing", error))
}

/// Rejects a surface schedule that rains on a cell whose support a selected source owns.
pub(in super::super) fn check_schedule(
    definition: &SurfaceChannelDefinition,
    schedule: &ForcingSchedule,
) -> Result<(), SurfaceChannelError> {
    match &definition.runoff {
        Some(runoff) => runoff
            .plan
            .check_surface_schedule(schedule)
            .map_err(|error| plan_error("suppressed-rain: surface forcing", error)),
        None => Ok(()),
    }
}

/// Rejects a trial whose runoff drains into surface cells, which cannot step yet.
pub(in super::super) fn check_steppable(
    definition: &SurfaceChannelDefinition,
) -> Result<(), SurfaceChannelError> {
    let Some(runoff) = &definition.runoff else {
        return Ok(());
    };
    let outlets = definition
        .setup()
        .runoff
        .as_ref()
        .map_or(&[][..], |setup| setup.outlets.as_slice());
    for lane in &runoff.lanes {
        if let RunoffTarget::Surface { cell } = lane.target {
            let id = outlets
                .get(lane.outlet)
                .map_or("", |o| o.outlet_id.as_str());
            return Err(SurfaceChannelError::input(format!(
                "runoff-attachment: runoff outlet={id:?} drains into surface cell={cell}, which cannot step yet: route the outlet into a reach or a pool"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelRunoffDestination;
    use super::super::scenario::{
        POND, SOURCE, build, forcing, input_message, outlet, parts, pond_parts, record, rejection,
        share, starts,
    };
    use crate::channel::ChannelLateralInput;
    use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult, client};
    use crate::forcing::WaterAmount;
    use crate::runoff::CellRainAssignment;
    use crate::solver::{ForcingInterval, ForcingSchedule, SolverSetup};
    use crate::storage::LevelPoolSources;

    /// One rainy interval over `[0, 1000)` on the surface of `surface`.
    fn rainy(surface: &SolverSetup) -> TestResult<ForcingSchedule> {
        Ok(ForcingSchedule::new(
            &surface.domain,
            &surface.boundaries,
            vec![ForcingInterval {
                start: 0.0,
                end: 1000.0,
                rain: vec![1e-6],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            }],
        )?)
    }

    #[test]
    fn a_surface_cell_destination_rejects_at_every_trial() -> TestResult {
        let (mut setup, initial) = parts()?;
        if let Some(runoff) = setup.runoff.as_mut() {
            runoff.outlets = vec![outlet(SurfaceChannelRunoffDestination::SurfaceCells {
                cells: vec![share(0, 1.0)],
            })];
        }
        let batch = build((setup, initial))?;
        let message = input_message(
            batch.trial_to(0.5, &forcing(&batch, 0.5, vec![record(0.0, 0.5, 1e-5)])),
        )?;
        starts(&message, "runoff-attachment");
        assert!(message.contains("surface cell=0"), "{message}");
        let reach = build(parts()?)?;
        let next = reach.trial_to(0.5, &forcing(&reach, 0.5, vec![record(0.0, 0.5, 1e-5)]))?;
        assert_eq!(next.diagnostics()?.runoff.len(), 1);
        Ok(())
    }

    #[test]
    fn runoff_records_reject_through_the_plan() -> TestResult {
        let batch = build(parts()?)?;
        let missing = forcing(&batch, 0.5, Vec::new());
        starts(
            &input_message(batch.trial_to(0.5, &missing))?,
            "runoff-forcing",
        );
        let shifted = forcing(&batch, 0.5, vec![record(0.0, 0.25, 0.0)]);
        starts(
            &input_message(batch.trial_to(0.5, &shifted))?,
            "runoff-forcing",
        );
        let mut runon = record(0.0, 0.5, 0.0);
        runon.input.external_runon_volumes_m3[0] = 1e-3;
        let message = input_message(batch.trial_to(0.5, &forcing(&batch, 0.5, vec![runon])))?;
        starts(&message, "runoff-forcing");
        assert!(message.contains("runon"), "{message}");
        let mut twice = forcing(&batch, 0.5, vec![record(0.0, 0.5, 0.0)]);
        twice.runoff.push(record(0.0, 0.5, 0.0));
        starts(
            &input_message(batch.trial_to(0.5, &twice))?,
            "runoff-forcing",
        );
        Ok(())
    }

    #[test]
    fn runoff_records_and_states_without_a_runoff_setup_reject() -> TestResult {
        let batch = build(pond_parts(false)?)?;
        let records = forcing(&batch, 0.5, vec![record(0.0, 0.5, 0.0)]);
        starts(
            &input_message(batch.trial_to(0.5, &records))?,
            "runoff-without-setup",
        );
        let (setup, mut initial) = pond_parts(false)?;
        initial.runoff = parts()?.1.runoff;
        starts(
            &input_message(build((setup, initial)))?,
            "runoff-without-setup",
        );
        Ok(())
    }

    #[test]
    fn caller_lateral_inputs_reject_the_reserved_prefix() -> TestResult {
        let batch = build(parts()?)?;
        let mut reserved = forcing(&batch, 0.5, vec![record(0.0, 0.5, 0.0)]);
        reserved.reaches[0]
            .lateral_inputs
            .push(ChannelLateralInput::Inflow {
                boundary_id: "runoff-attachment:caller".into(),
                amount: WaterAmount::Rate { rate_m3_per_s: 0.0 },
                longitudinal_velocity_m_per_s: 0.0,
            });
        starts(
            &input_message(batch.trial_to(0.5, &reserved))?,
            "reserved-binding-prefix",
        );
        Ok(())
    }

    #[test]
    fn suppressed_cell_rain_rejects_at_new_continue_and_resume() -> TestResult {
        let (mut setup, initial) = parts()?;
        let wet = rainy(&setup.surface)?;
        setup.surface.forcing = wet.clone();
        starts(&rejection(setup)?, "suppressed-rain");
        let batch = build((parts()?.0, initial))?;
        starts(
            &input_message(batch.continue_surface_forcing(wet.clone()))?,
            "suppressed-rain",
        );
        let data = batch.snapshot_data()?;
        let resumed = Batch::resume(client(), &data, wet);
        starts(&input_message(resumed)?, "suppressed-rain");
        Ok(())
    }

    #[test]
    fn rain_on_a_pool_owned_by_a_selected_source_rejects() -> TestResult {
        let (mut setup, mut initial) = pond_parts(true)?;
        let (runoff_setup, runoff_initial) = parts()?;
        let mut runoff = runoff_setup.runoff.ok_or("missing runoff")?;
        runoff.support.contributions[0]
            .atoms
            .push("pond-surface".into());
        runoff.support.pool_rain[0].rain = CellRainAssignment::Suppressed {
            selected_contribution_id: SOURCE.into(),
        };
        runoff.definitions = vec![super::super::scenario::definition(5.0)?];
        setup.runoff = Some(runoff);
        initial.runoff = runoff_initial.runoff;
        let batch = build((setup, initial))?;
        let sources = |rain| {
            Some(vec![LevelPoolSources {
                rain_rate_m_per_s: rain,
                potential_evaporation_rate_m_per_s: 0.0,
                bottom_loss_rate_m3_per_s: 0.0,
            }])
        };
        let mut wet = forcing(&batch, 0.5, vec![record(0.0, 0.5, 0.0)]);
        wet.pool_sources = sources(1e-6);
        let message = input_message(batch.trial_to(0.5, &wet))?;
        starts(&message, "suppressed-rain");
        assert!(message.contains(POND), "{message}");
        let mut dry = wet;
        dry.pool_sources = sources(0.0);
        batch.trial_to(0.5, &dry)?;
        Ok(())
    }
}
