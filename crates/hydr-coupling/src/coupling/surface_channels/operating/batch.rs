//! The coordinator's side of its operating children: interval ends, attempt operations, and
//! completion with the candidate.

use super::{AttemptOperation, CandidateNodes, OperatingChildren, StepOperation};
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::batch::SurfaceChannelBatch;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};
use std::sync::Arc;

/// Next operating children and the interval operation of an accepted candidate.
pub(in super::super) type Completed<R> =
    (Option<OperatingChildren<R>>, Option<Arc<StepOperation<R>>>);

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    /// The trial end `limited`, shortened to the next operating event after the accepted time.
    ///
    /// No interval then contains an event. A retry only shortens it, so no retry crosses one.
    pub(in super::super) fn operating_end(&self, limited: f64) -> Result<f64, SurfaceChannelError> {
        let mut end = match &self.operating {
            Some(children) => limited.min(children.next_event_after(self.time_s)?),
            None => limited,
        };
        if let Some(event) = super::super::geometry::next_event(
            &self.definition.setup().geometry_schedules,
            self.time_s,
        ) {
            end = end.min(event);
        }
        if let Some(event) = super::super::geometry::next_pool_curve_event(
            &self.definition.setup().pool_curve_schedules,
            self.time_s,
        ) {
            end = end.min(event);
        }
        if let Some(event) = super::super::geometry::next_surface_bed_event(
            &self.definition.setup().surface_bed_schedules,
            self.time_s,
        ) {
            end = end.min(event);
        }
        if let Some(event) = super::super::geometry::next_channel_section_event(
            &self.definition.setup().channel_section_schedules,
            self.time_s,
        ) {
            end = end.min(event);
        }
        Ok(end)
    }

    /// Link laws and prepared operating children of the attempt ending at `end`.
    pub(in super::super) fn operation_at(
        &self,
        end: f64,
    ) -> Result<AttemptOperation<R>, SurfaceChannelError> {
        let mut attempt = match &self.operating {
            Some(children) => children.prepare(&self.context, self.time_s, end)?,
            None => AttemptOperation::declared(&self.context, end),
        };
        let schedules = &self.definition.setup().geometry_schedules;
        if !schedules.is_empty() {
            let points = attempt.operations.points.clone();
            let (operations, geometry) = super::super::geometry::evaluate_interval(
                &self.context,
                schedules,
                points,
                self.time_s,
                end,
            )?;
            attempt.operations = operations;
            attempt.geometry = geometry;
        }
        Ok(attempt)
    }

    /// Completes the operating children of `operation` at `end` from the candidate `nodes`:
    /// its end stages and end wet masks at every donor group.
    ///
    /// The trial calls it once the candidate passes every other check, so a retryable rejection
    /// still shortens the interval before any decision samples it.
    pub(in super::super) fn complete_operating(
        &self,
        operation: AttemptOperation<R>,
        end: f64,
        nodes: Option<&(Tensor<R>, Tensor<R>)>,
    ) -> Result<Completed<R>, SurfaceChannelError> {
        match (&self.operating, operation.prepared) {
            (Some(children), Some(prepared)) => {
                let candidate = nodes.map(|(stage, wet)| CandidateNodes { stage, wet });
                let points = &operation.operations.points;
                let (next, step) = children.complete(prepared, end, candidate, points)?;
                Ok((Some(next), Some(Arc::new(step))))
            }
            (None, None) => Ok((None, None)),
            _ => Err(SurfaceChannelError::Numerical(
                "the attempt's operation differs from the operating children".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::SurfaceChannelOperating;
    use super::super::scenario::{build, controlled_gate, signal};
    use crate::coupling::surface_channels::pools::scenario::{
        PoolForcing, advance, check, declaration, inflow_binding, inflow_input, link, pond, pool,
    };
    use crate::coupling::surface_channels::structures::SurfaceChannelOperationOwner;
    use crate::coupling::surface_channels::structures::scenario::{
        Batch, Tanks, TestResult, gate_law,
    };
    use crate::structures::{DirectionPolicy, StructureOperatingPoint};

    /// Decision times of the pond relay, in s.
    const DECISIONS: [f64; 12] = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    /// Run end, in s.
    const END_S: f64 = 12.0;
    /// Requested duration of every interval, in s.
    const DT: f64 = 0.5;
    /// Pond inflow, in m³/s.
    const FEED: f64 = 0.2;
    /// Target opening of the outlet, in m.
    const TARGET_M: f64 = 0.2;
    /// Smallest admitted common step of the tanks, in s.
    const MINIMUM_STEP_S: f64 = 1e-10;

    /// A 4 m² pond at 1 m, fed at `FEED`, with an actuated outlet at `rate` onto a 100 m² basin.
    ///
    /// The relay opens the outlet at the decision after the pond passes 1.125 m, at 3 s.
    fn pond_outlet(rate: f64) -> TestResult<Batch> {
        let mut outlet = link(
            "outlet",
            gate_law(0.6)?,
            DirectionPolicy::Bidirectional,
            StructureOperatingPoint::GateOpening { opening_m: 0.0 },
            (pool("pond"), 0.0),
            (pool("basin"), 0.0),
        );
        outlet.operation = SurfaceChannelOperationOwner::Actuated;
        let operating = SurfaceChannelOperating {
            gates: Some(controlled_gate(
                "outlet",
                [0.7, 1.125],
                (TARGET_M, rate),
                (0.0, &DECISIONS, 20.0),
            )),
            pumps: None,
            trajectories: Vec::new(),
            signals: vec![signal("outlet", pool("pond"))],
        };
        let pools = declaration(
            vec![
                pond("basin", 0.0, 100.0, 4.0)?,
                pond("pond", 0.0, 4.0, 4.0)?,
            ],
            false,
            vec![inflow_binding("feed", "pond")],
        );
        Ok(build(
            Tanks::at(0.0, 0.1),
            pools,
            &[20.0, 4.0],
            vec![outlet],
            operating,
        )?)
    }

    /// Pending arrival window `[lower, upper]` of the accepted outlet, if any.
    fn window(batch: &Batch) -> TestResult<Option<(f64, f64)>> {
        let operating = batch.operating.as_ref().ok_or("operating children")?;
        let gate = operating.gate().ok_or("the gate child")?;
        let output = gate.actuators().outputs()?.remove(0);
        Ok(output.arrival_lower_s.zip(output.arrival_upper_s))
    }

    /// Accepted arrival count of the outlet actuator.
    fn arrivals(batch: &Batch) -> TestResult<usize> {
        let operating = batch.operating.as_ref().ok_or("operating children")?;
        let gate = operating.gate().ok_or("the gate child")?;
        Ok(gate.actuators().states()?[0].arrival_count)
    }

    /// The outlet arrives at its target inside the run at actuator `rate`. Each rate of
    /// `0.05`, `0.1`, and `0.025` m/s lands the exact arrival on a decision time. `0.06` m/s
    /// lands it between decisions.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval to 12 s accepts and lasts at least `minimum_step_s`.
    /// - Every interval passes its mass bound, every pairing, and the composite balance.
    /// - Every decision time ends an interval, and no interval holds one strictly inside it.
    /// - No interval holds a pending arrival window `[l, u]` with `start < l` and `u < end`.
    /// - The actuator arrives at least once inside the run.
    fn arrives_inside_the_run(rate: f64) -> TestResult {
        let mut batch = pond_outlet(rate)?;
        let forcing = PoolForcing {
            sources: None,
            inflows: vec![inflow_input("feed", FEED)],
        };
        let mut ends = Vec::new();
        while batch.time() < END_S {
            let start = batch.time();
            let pending = window(&batch)?;
            batch = advance(&batch, DT.min(END_S - start), &forcing)?;
            let end = batch.time();
            ends.push(end);
            assert!(
                end - start >= MINIMUM_STEP_S,
                "rate={rate}: [{start},{end}]"
            );
            check(&batch, &[0.0, FEED * (end - start)])?;
            for decision in DECISIONS {
                assert!(
                    !(start < decision && decision < end),
                    "rate={rate}: [{start},{end}] holds {decision}"
                );
            }
            if let Some((lower, upper)) = pending {
                assert!(
                    !(start < lower && upper < end),
                    "rate={rate}: [{start},{end}] holds [{lower},{upper}]"
                );
            }
        }
        assert!(DECISIONS.iter().all(|t| ends.contains(t)), "{ends:?}");
        assert!(arrivals(&batch)? >= 1, "rate={rate}: no arrival");
        Ok(())
    }

    #[test]
    fn an_arrival_between_decisions_takes_no_sliver() -> TestResult {
        arrives_inside_the_run(0.06)
    }

    #[test]
    fn an_arrival_on_a_decision_takes_no_sliver() -> TestResult {
        for rate in [0.05, 0.1, 0.025] {
            arrives_inside_the_run(rate)?;
        }
        Ok(())
    }
}
