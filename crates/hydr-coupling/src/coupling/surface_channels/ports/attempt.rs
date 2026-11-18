//! Port steps of one attempt: the surface completion without links and the port receipts.

use super::super::SurfaceChannelError;
use super::super::context::{Context, Ops};
use super::super::state::StoredStep;
use super::{PortBinding, PortInterval};
use crate::solver::Solver;
use hydr_solver::internal::{CouplingCompletion, SurfaceCouplingStage};
use numr::dtype::DType;
use numr::runtime::Runtime;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Completes the surface `stage` of a coordinator without links. Port water needs no exchange.
    pub fn complete_alone(
        &self,
        surface: &Solver<R>,
        stage: &SurfaceCouplingStage<R>,
    ) -> Result<Solver<R>, SurfaceChannelError> {
        match surface.coupling_complete(stage, None)? {
            CouplingCompletion::Accepted(next) => Ok(*next),
            CouplingCompletion::Retry(reason) => Err(SurfaceChannelError::Numerical(format!(
                "surface completion: {reason}. Shorten the common interval"
            ))),
        }
    }

    /// Port receipts of the accepted attempt of `binding` over `dt`. Empty without ports.
    ///
    /// `previous` is the receipt of the interval before, for the reversal flags.
    pub fn port_interval(
        &self,
        binding: &PortBinding<R>,
        dt: f64,
        previous: Option<&StoredStep<R>>,
    ) -> Result<PortInterval, SurfaceChannelError> {
        let Some(topology) = &self.ports else {
            return Ok(PortInterval::default());
        };
        PortInterval::new(
            &topology.ids,
            &binding.fluxes,
            binding.surface_rate_m3_per_s,
            dt,
            previous.map(|step| step.ports.transfers.as_slice()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::forcing::{BoundForcing, PortedEnds};
    use super::super::super::{SurfaceChannelBatch, SurfaceChannelDefinition};
    use super::super::PortInterval;
    use super::super::fixture::{ALPHA, BETA, TestResult};
    use super::super::receipt::PortRegime;
    use super::super::scenario::{
        Batch, CELLS, Reach, Scenario, client, east, forcing, receipt, step, still, west,
    };
    use crate::channel::ChannelCellResult;
    use hydr_solver::internal::CouplingPreparation;

    /// Requested duration of every test interval, in s. The CFL limit shortens it.
    const DT: f64 = 0.1;

    /// Accepted intervals of the seiche run.
    const SEICHE_INTERVALS: usize = 120;

    fn reach_cells(batch: &Batch, id: &str) -> TestResult<Vec<ChannelCellResult>> {
        Ok(batch.reach(id).ok_or("the reach exists")?.state()?)
    }

    /// Fails unless the lifetime composite balance holds.
    fn balanced(batch: &Batch) -> TestResult {
        let balance = batch.ledger()?;
        if balance.residual_m3.abs() <= balance.bound_m3 {
            Ok(())
        } else {
            Err(format!(
                "composite residual={}, bound={}",
                balance.residual_m3, balance.bound_m3
            )
            .into())
        }
    }

    #[test]
    fn a_lake_at_rest_through_ports_with_a_bed_step_stays_at_rest() -> TestResult {
        // Frozen criterion: after 50 intervals every |q|, |Q| and stage change is at most 1e-12.
        const TOLERANCE: f64 = 1e-12;
        let stage = 1.6;
        // The reach bed lies below, then above, the 1 m surface bed, under the 1.2 m bank crest.
        for alpha_bed in [0.5, 1.125] {
            let ports = vec![west("p-west"), east("p-east")];
            let mut batch = Scenario::rest(1.0, stage, alpha_bed, ports).build()?;
            for _ in 0..50 {
                batch = step(&batch, DT)?;
                for transfer in &receipt(&batch)?.transfers {
                    assert_eq!(transfer.regime, PortRegime::Subcritical);
                    assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
                }
            }
            assert_eq!(batch.accepted_steps(), 50);
            for cell in batch.surface().state()?.cells() {
                assert!(
                    cell.qx.abs() <= TOLERANCE && cell.qy.abs() <= TOLERANCE,
                    "{cell:?}"
                );
                assert!((cell.h + 1.0 - stage).abs() <= TOLERANCE, "{cell:?}");
            }
            for id in [ALPHA, BETA] {
                for cell in reach_cells(&batch, id)? {
                    assert!(cell.discharge_m3_per_s.abs() <= TOLERANCE, "{id}: {cell:?}");
                    let moved = (cell.stage_m.ok_or("the reach is wet")? - stage).abs();
                    assert!(moved <= TOLERANCE, "{id}: {cell:?}");
                }
            }
            balanced(&batch)?;
        }
        Ok(())
    }

    #[test]
    fn water_below_the_higher_bed_moves_nothing() -> TestResult {
        // The reach stage of 0.8 m lies below the 1 m surface bed, and the surface is dry.
        let mut case = Scenario::rest(1.0, 0.8, 0.5, vec![west("p")]);
        case.link = false;
        let batch = step(&case.build()?, DT)?;
        let transfer = &receipt(&batch)?.transfers[0];
        assert_eq!(transfer.regime, PortRegime::Dry);
        assert_eq!(transfer.signed_volume_m3.to_bits(), 0.0f64.to_bits());
        assert_eq!(
            batch.surface().ledger().exchange_in_m3.to_bits(),
            0.0f64.to_bits()
        );
        Ok(())
    }

    #[test]
    fn a_port_only_coordinator_builds_and_steps() -> TestResult {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.link = false;
        case.alpha = Reach::rest(0.5, 1.8);
        let mut batch = case.build()?;
        assert!(batch.context.links.is_none() && batch.context.link_count == 0);
        for _ in 0..5 {
            batch = step(&batch, DT)?;
            let interval = receipt(&batch)?;
            // The higher reach drains onto the surface.
            assert!(interval.transfers[0].signed_volume_m3 < 0.0);
            assert!(interval.surface_m3[0] > 0.0 && interval.reach_m3[1] > 0.0);
            balanced(&batch)?;
        }
        let diagnostics = batch.last_step()?.ok_or("an accepted interval")?;
        assert!(diagnostics.links.is_empty());
        assert_eq!(diagnostics.iterations, 0);
        assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
        let counters = &batch.counters.ports;
        assert!(counters.to_surface_m3[0] > 0.0 && counters.to_channel_m3[0] == 0.0);
        // The link totals the balance reports leave port water out.
        assert!(batch.ledger()?.surface_exchange_in_m3.abs() <= 1e-12);
        Ok(())
    }

    #[test]
    fn a_draining_end_cell_shortens_the_attempt_and_keeps_its_water_nonnegative() -> TestResult {
        // Only the reach end cell holds water, and the floodplain is dry.
        let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
        case.link = false;
        case.alpha = Reach {
            bed: 1.0,
            depth: [0.05, 0.0, 0.0],
            discharge_per_width: [0.0; 3],
        };
        let start = case.build()?;
        let held = reach_cells(&start, ALPHA)?[0].volume_m3;
        let mut batch = step(&start, 1.0)?;
        let first = batch.last_step()?.ok_or("an accepted interval")?;
        assert!(first.retries >= 1, "the requested second must shorten");
        // The first interval drains across the front. Later the floodplain cell is wet.
        assert_eq!(
            receipt(&batch)?.transfers[0].regime,
            PortRegime::WetDryFront
        );
        let mut drained = 0.0;
        for _ in 0..10 {
            let transfer = &receipt(&batch)?.transfers[0];
            drained -= transfer.signed_volume_m3;
            let reach = batch.reach(ALPHA).ok_or("alpha")?;
            let limits = reach.last_step()?.ok_or("a reach receipt")?;
            // Port export enters the face depletion limit, which covers the interval.
            let dt = limits.end_s - limits.start_s;
            assert!(limits.depletion_limit_s.is_finite() && dt <= limits.depletion_limit_s);
            assert!(dt <= limits.cfl_limit_s);
            assert!(
                reach_cells(&batch, ALPHA)?
                    .iter()
                    .all(|c| c.volume_m3 >= 0.0)
            );
            assert!(batch.surface().state()?.cells().iter().all(|c| c.h >= 0.0));
            balanced(&batch)?;
            batch = step(&batch, 1.0)?;
        }
        assert!(drained > 0.0 && drained <= held);
        Ok(())
    }

    #[test]
    fn a_dam_break_onto_a_dry_floodplain_runs_and_balances() -> TestResult {
        // Ritter smoke case: a still reach releases onto a dry floodplain through its port.
        let mut case = Scenario::rest(1.0, 0.0, 1.0, vec![west("p")]);
        case.link = false;
        case.alpha = Reach::uniform(1.0, 0.5, 0.0);
        let mut batch = case.build()?;
        let held: f64 = reach_cells(&batch, ALPHA)?
            .iter()
            .map(|c| c.volume_m3)
            .sum();
        let mut released = 0.0;
        for index in 0..20 {
            batch = step(&batch, DT)?;
            let transfer = &receipt(&batch)?.transfers[0];
            if index == 0 {
                assert_eq!(transfer.regime, PortRegime::WetDryFront);
            }
            assert!(transfer.signed_volume_m3 <= 0.0);
            released -= transfer.signed_volume_m3;
            assert!(
                reach_cells(&batch, ALPHA)?
                    .iter()
                    .all(|c| c.volume_m3 >= 0.0)
            );
            assert!(batch.surface().state()?.cells().iter().all(|c| c.h >= 0.0));
            balanced(&batch)?;
        }
        assert!(released > 0.0 && released < held);
        Ok(())
    }

    #[test]
    fn a_shared_end_cell_donor_reserves_its_port_export() -> TestResult {
        // The bank link and the port both draw on `alpha` cell 0.
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.alpha = Reach::rest(0.5, 1.75);
        let (mut setup, initial) = case.parts()?;
        setup.links[0].channel_cell = 0;
        setup.links[0].surface_cell = 0;
        let definition = SurfaceChannelDefinition::with_ports(setup, case.ports.clone())?;
        let batch = SurfaceChannelBatch::from_definition(client(), definition, initial)?;
        let context = &batch.context;
        let ids = [ALPHA, BETA];
        let dt = 0.05;
        let bound = BoundForcing::bind(
            context,
            &ids,
            (0.0, dt),
            &forcing(0.0, dt),
            PortedEnds::Coordinator,
        )?;
        let start = context.start(&batch.surface, &batch.reaches, &ids, dt, bound)?;
        // The linked reach is ported, so each attempt builds its slot operands.
        assert!(start.slots.is_none() && start.reservations[0].is_none());
        let topology = context.ports.as_ref().ok_or("ports")?;
        let ghosts = &start.ports.as_ref().ok_or("ghosts")?.ghosts;
        let CouplingPreparation::Ready(stage) = batch.surface.coupling_prepare(
            dt,
            0,
            context.links.as_ref(),
            Some((&topology.surface, ghosts)),
        )?
        else {
            return Err("the stage must prepare".into());
        };
        let ports = context.bind_ports(&stage, &start, &batch.reaches, &ids)?;
        let export = ports.fluxes[0].outward_water_rate_m3_per_s;
        assert!(export > 0.0);
        let reservation = ports.reservation(&start, 0)?;
        let donor = reservation.donor_volume_m3.try_to_vec::<f64>()?[0];
        // The still interior face exports nothing, so the end cell exports its port water only.
        let outgoing = reservation.outgoing_rate_m3_per_s.try_to_vec::<f64>()?[0];
        assert_eq!(outgoing.to_bits(), export.to_bits());
        let attempt = context.attempt(&start, *stage, ports)?;
        let available = attempt.available.try_to_vec::<f64>()?[context.surface_count];
        let margin = 4.0 * f64::EPSILON * (context.slot_withdrawals[0] as f64 + 8.0);
        let expected = (donor * (1.0 - margin) - (outgoing * dt) * (margin + 1.0)).max(0.0);
        assert_eq!(available.to_bits(), expected.to_bits());
        assert!(available < donor - export * dt);
        // The full interval passes both pairings and the combined depletion.
        let next = step(&batch, dt)?;
        assert!(receipt(&next)?.transfers[0].signed_volume_m3 < 0.0);
        let diagnostics = next.last_step()?.ok_or("an accepted interval")?;
        assert!(diagnostics.links[0].signed_volume_m3 < 0.0);
        assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
        balanced(&next)
    }

    /// The reach starts above the surface. Water runs out through the port, then back.
    ///
    /// Every interval checks the step mass bound and the lifetime composite balance.
    /// Returns the port receipts of every interval and the last batch.
    fn seiche(intervals: usize) -> TestResult<(Vec<PortInterval>, Batch)> {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.link = false;
        case.alpha = Reach::rest(0.5, 1.75);
        let mut batch = case.build()?;
        let mut run = Vec::with_capacity(intervals);
        for _ in 0..intervals {
            batch = step(&batch, 0.25)?;
            let diagnostics = batch.last_step()?.ok_or("an accepted interval")?;
            assert!(diagnostics.mass_residual_m3.abs() <= diagnostics.mass_bound_m3);
            balanced(&batch)?;
            run.push(receipt(&batch)?.clone());
        }
        Ok((run, batch))
    }

    #[test]
    fn pairing_and_composite_balance_hold_through_inflow_and_outflow() -> TestResult {
        let (run, last) = seiche(SEICHE_INTERVALS)?;
        let (mut inflow, mut outflow) = (false, false);
        for interval in &run {
            for transfer in &interval.transfers {
                assert!(transfer.paired_gap_m3.abs() <= transfer.paired_bound_m3);
                inflow |= transfer.signed_volume_m3 > 0.0;
                outflow |= transfer.signed_volume_m3 < 0.0;
            }
        }
        assert!(inflow && outflow, "the seiche must run both ways");
        let counters = &last.counters.ports;
        assert!(counters.to_channel_m3[0] > 0.0 && counters.to_surface_m3[0] > 0.0);
        Ok(())
    }

    #[test]
    fn a_reversal_sets_the_reversed_flag() -> TestResult {
        let (run, _) = seiche(SEICHE_INTERVALS)?;
        let mut previous: Option<f64> = None;
        let mut reversals = 0;
        for interval in &run {
            let transfer = &interval.transfers[0];
            let signed = transfer.signed_volume_m3;
            let flipped =
                previous.is_some_and(|p| (p > 0.0 && signed < 0.0) || (p < 0.0 && signed > 0.0));
            assert_eq!(transfer.reversed, flipped);
            reversals += usize::from(flipped);
            previous = Some(signed);
        }
        assert!(reversals >= 1, "the seiche must reverse the port");
        Ok(())
    }

    #[test]
    fn a_portless_coordinator_matches_its_public_trial_bit_for_bit() -> TestResult {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, Vec::new());
        case.alpha = Reach::uniform(0.5, 1.25, 0.4);
        case.surface = (0..CELLS)
            .map(|cell| still(0.4 + 0.05 * cell as f64))
            .collect();
        let (setup, initial) = case.parts()?;
        let mut public = SurfaceChannelBatch::new(client(), setup, initial)?;
        let mut owned = case.build()?;
        for _ in 0..5 {
            let start = public.time();
            let end = start + 0.25;
            public = public.trial_to(end, &forcing(start, end))?.into_next();
            owned = step(&owned, 0.25)?;
            let bits = |batch: &Batch| -> TestResult<Vec<u64>> {
                let mut bits: Vec<u64> = batch
                    .surface()
                    .state()?
                    .cells()
                    .iter()
                    .flat_map(|cell| [cell.h, cell.qx, cell.qy])
                    .map(f64::to_bits)
                    .collect();
                for id in [ALPHA, BETA] {
                    for cell in reach_cells(batch, id)? {
                        bits.push(cell.volume_m3.to_bits());
                        bits.push(cell.longitudinal_momentum_m4_per_s.to_bits());
                    }
                }
                bits.push(batch.time().to_bits());
                Ok(bits)
            };
            assert_eq!(bits(&public)?, bits(&owned)?);
            assert_eq!(public.last_step()?, owned.last_step()?);
            assert!(receipt(&owned)?.transfers.is_empty());
        }
        Ok(())
    }
}
