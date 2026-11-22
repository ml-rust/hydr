//! Port steps of one attempt: the surface completion without links and the port receipts.

use super::super::SurfaceChannelError;
use super::super::context::{Context, Ops};
use super::{PortBinding, PortCounters, PortInterval};
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
    /// `counters` holds each port's signed volume of the interval before, for the reversal flags.
    pub fn port_interval(
        &self,
        binding: &PortBinding<R>,
        dt: f64,
        counters: &PortCounters,
    ) -> Result<PortInterval, SurfaceChannelError> {
        let Some(topology) = &self.ports else {
            return Ok(PortInterval::default());
        };
        PortInterval::new(
            &topology.ids,
            &binding.fluxes,
            binding.surface_rate_m3_per_s,
            dt,
            &counters.last_signed_m3,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::super::SurfaceChannelBatch;
    use super::super::super::forcing::BoundForcing;
    use super::super::fixture::{ALPHA, BETA, TestResult};
    use super::super::scenario::{Batch, Reach, Scenario, client, forcing, receipt, step, west};
    use hydr_solver::internal::CouplingPreparation;

    /// Requested duration of every test interval, in s. The CFL limit shortens it.
    const DT: f64 = 0.1;

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
    fn a_port_only_coordinator_holds_no_link_operands() -> TestResult {
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.link = false;
        case.alpha = Reach::rest(0.5, 1.8);
        let batch = step(&case.build()?, DT)?;
        assert!(batch.context.links.is_none() && batch.context.operations.structures().is_err());
        assert_eq!(batch.context.link_count, 0);
        let counters = &batch.counters.ports;
        // The higher reach drains onto the surface. Both sides count the port water.
        assert!(counters.to_surface_m3[0] > 0.0 && counters.to_channel_m3[0] == 0.0);
        assert!(counters.surface_m3[0] > 0.0 && counters.surface_m3[1] == 0.0);
        balanced(&batch)
    }

    #[test]
    fn a_shared_end_cell_donor_reserves_its_port_export() -> TestResult {
        // The bank link and the port both draw on `alpha` cell 0.
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.alpha = Reach::rest(0.5, 1.75);
        let (mut setup, initial) = case.parts()?;
        setup.links[0].channel_cell = 0;
        setup.links[0].surface_cell = 0;
        let batch = SurfaceChannelBatch::new(client(), setup, initial)?;
        let context = &batch.context;
        let ids = [ALPHA, BETA];
        let dt = 0.05;
        let held = forcing(&batch.setup().ports, 0.0, dt);
        let bound = BoundForcing::bind(context, &ids, (0.0, dt), &held)?;
        let children = (&batch.surface, &batch.reaches[..], None);
        let start = context.start(children, &ids, (dt, &context.operations), bound)?;
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
        let attempt = context.attempt(
            &start,
            (*stage, std::sync::Arc::clone(&context.operations)),
            ports,
            None,
            None,
        )?;
        let available = attempt
            .available
            .as_ref()
            .ok_or("a reach without pool ends reserves per attempt")?
            .try_to_vec::<f64>()?[context.surface_count];
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

    #[test]
    fn the_counters_carry_the_reversal_reference_across_intervals() -> TestResult {
        // The reach starts above the surface, so water runs out through the port, then back.
        let mut case = Scenario::rest(1.0, 1.6, 0.5, vec![west("p")]);
        case.alpha = Reach::rest(0.5, 1.75);
        let mut batch = case.build()?;
        for _ in 0..40 {
            let before = batch.counters.ports.last_signed_m3[0];
            batch = step(&batch, 0.25)?;
            let transfer = &receipt(&batch)?.transfers[0];
            let signed = transfer.signed_volume_m3;
            let flipped = (before > 0.0 && signed < 0.0) || (before < 0.0 && signed > 0.0);
            assert_eq!(transfer.reversed, flipped);
            assert_eq!(
                batch.counters.ports.last_signed_m3[0].to_bits(),
                signed.to_bits()
            );
        }
        Ok(())
    }
}
