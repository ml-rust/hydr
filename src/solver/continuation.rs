//! Forecast continuation under forcing spliced after the accepted history.

use super::compatibility::same_interval;
use super::definition::{Definition, retained_index};
use super::forcing_history::{self, ForcingPrefix};
use super::receipts::{StepReceipt, rebase_forcing_index};
use super::{EngineError, ForcingInterval, ForcingSchedule, SnapshotData, Solver};
use crate::tensor::TensorDomain;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + TypeConversionOps<R>
        + HypotOps<R>
        + numr::ops::CumulativeOps<R>
        + numr::ops::ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Splices `forcing` after the accepted history.
    ///
    /// Keeps accepted state, ledger, budget, step count and last step.
    /// Checks everything before changing anything.
    ///
    /// Every interval that starts before the accepted time `t` is frozen, end included.
    /// `forcing` starts at or before `t`, at a retained interval start or at `t` on a boundary.
    /// It reproduces every frozen interval it covers bit for bit, provenance included.
    /// A schedule that starts before the retained window folds to the engine's forcing prefix.
    /// The engine then keeps only the window that rule W retains, on the host and on the device.
    ///
    /// # Errors
    /// Rejects a rewritten frozen interval, a start that is no retained boundary, an altered
    /// folded prefix, coverage that ends before `t`, or forcing that fails `ForcingSchedule::new`
    /// for this engine's domain and boundaries.
    pub fn continue_with_forcing(&mut self, forcing: ForcingSchedule) -> Result<(), EngineError> {
        let time = self.accepted.time;
        let last_start = self.accepted.last.as_ref().map(StepReceipt::start);
        let (combined, shared) = splice(&self.definition, time, forcing)?;
        let retained = retained_index(combined.intervals(), time, last_start);
        let definition = self.definition.continued(combined, retained, shared)?;
        let context = TensorDomain::<R>::new(
            self.context.client.clone(),
            &definition.domain,
            &definition.forcing,
        )?;
        let mut last = self.accepted.last.clone();
        if let Some(StepReceipt::Device(receipt)) = &mut last {
            receipt.forcing_index = rebase_forcing_index(
                receipt.forcing_index,
                self.definition.prefix.intervals,
                definition.prefix.intervals,
                context.sources.len(),
            )?;
        }
        self.definition = Arc::new(definition);
        self.context = Arc::new(context);
        self.accepted.last = last;
        Ok(())
    }

    /// Builds an engine from an owned package and continues it under `forcing`.
    ///
    /// The engine follows `data.forcing_prefix` with the package window, then imports the package.
    /// `data.setup.forcing.clone()` as `forcing` is a plain restart.
    /// See [`Solver::continue_with_forcing`] for the splice rule.
    ///
    /// # Errors
    /// Returns construction, import, and continuation errors.
    pub fn resume(
        client: R::Client,
        data: &SnapshotData,
        forcing: ForcingSchedule,
    ) -> Result<Self, EngineError> {
        let mut engine = Self::with_prefix(
            client,
            data.setup.clone(),
            Some(data.forcing_prefix.clone()),
            data.accepted.clone(),
        )?;
        engine.restore_data(data)?;
        engine.continue_with_forcing(forcing)?;
        Ok(engine)
    }
}

/// Returns the checked spliced schedule after the prefix and its bit-identical leading length.
fn splice(
    definition: &Definition,
    time: f64,
    forcing: ForcingSchedule,
) -> Result<(ForcingSchedule, usize), EngineError> {
    let old = definition.forcing.intervals();
    let new = forcing.intervals();
    let prefix = &definition.prefix;
    let start = new.first().map_or(f64::NAN, |interval| interval.start);
    if start.is_nan() || start > time {
        return Err(EngineError::rejected(
            "continuation forcing",
            format!(
                "start={start}, accepted time={time}: start the schedule at or before the accepted time"
            ),
        ));
    }
    let (kept, tail) = if start < prefix.end {
        (0, after_prefix(prefix, new)?)
    } else {
        let at_start = old
            .iter()
            .position(|interval| interval.start.to_bits() == start.to_bits());
        let at_end = old
            .last()
            .filter(|interval| interval.end.to_bits() == start.to_bits())
            .map(|_| old.len());
        let kept = at_start.or(at_end).ok_or_else(|| {
            EngineError::rejected(
                "continuation forcing",
                format!(
                    "start={start}, retained window [{}, {}): start the schedule at a retained interval start, or at the accepted time on an interval boundary",
                    prefix.end,
                    old.last().map_or(prefix.end, |interval| interval.end)
                ),
            )
        })?;
        (kept, new)
    };
    let frozen = old.partition_point(|interval| interval.start < time);
    for (local, previous) in old.iter().enumerate().take(frozen).skip(kept) {
        let index = prefix.intervals.saturating_add(local);
        match tail.get(local - kept) {
            Some(next) if same_interval(previous, next) => {}
            Some(next) => {
                return Err(EngineError::rejected(
                    "continuation forcing",
                    format!(
                        "frozen interval {index} [{}, {}) differs from supplied [{}, {}) at accepted time {time}: repeat every interval that starts before the accepted time bit for bit, provenance included",
                        previous.start, previous.end, next.start, next.end
                    ),
                ));
            }
            None => {
                return Err(EngineError::rejected(
                    "continuation forcing",
                    format!(
                        "frozen interval {index} [{}, {}) is missing at accepted time {time}: supply forcing that covers the accepted time",
                        previous.start, previous.end
                    ),
                ));
            }
        }
    }
    // The checks above end the borrow of `forcing`; its new tail then moves, never clones.
    let skipped = new.len() - tail.len();
    let mut supplied = forcing.into_intervals();
    let mut combined = Vec::with_capacity(kept + supplied.len() - skipped);
    combined.extend_from_slice(&old[..kept]);
    combined.extend(supplied.drain(skipped..));
    let combined = ForcingSchedule::new(&definition.domain, &definition.boundaries, combined)?;
    let end = combined
        .intervals()
        .last()
        .map_or(f64::NAN, |interval| interval.end);
    if end.is_nan() || end < time {
        return Err(EngineError::rejected(
            "continuation forcing",
            format!(
                "end={end}, accepted time={time}: supply forcing that covers the accepted time"
            ),
        ));
    }
    Ok((combined, frozen))
}

/// Returns the intervals after the prefix base once those before it fold to `prefix` exactly.
fn after_prefix<'a>(
    prefix: &ForcingPrefix,
    new: &'a [ForcingInterval],
) -> Result<&'a [ForcingInterval], EngineError> {
    let count = new
        .iter()
        .position(|interval| interval.end.to_bits() == prefix.end.to_bits())
        .map(|index| index + 1);
    let folded = count.zip(new.first()).and_then(|(count, first)| {
        forcing_history::folded(&ForcingPrefix::empty(first.start), new, None, count)
    });
    match (count, folded) {
        (Some(count), Some(folded))
            if folded.intervals == prefix.intervals
                && folded.start.to_bits() == prefix.start.to_bits()
                && folded.end.to_bits() == prefix.end.to_bits()
                && folded.digest == prefix.digest =>
        {
            Ok(&new[count..])
        }
        _ => Err(EngineError::rejected(
            "continuation forcing prefix",
            format!(
                "supplied intervals before {} differ from the folded prefix of {} intervals from {}: supply the original intervals bit for bit with one ending exactly at {}",
                prefix.end, prefix.intervals, prefix.start, prefix.end
            ),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandModel, SoilState};
    use crate::solver::{
        BoundaryConditions, InitialState, SolverConfig, SolverSetup, StepDiagnostics,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};

    fn intervals(count: usize) -> Vec<ForcingInterval> {
        (0..count)
            .map(|k| ForcingInterval {
                start: k as f64,
                end: (k + 1) as f64,
                rain: vec![1e-6 * (k + 1) as f64],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn continuation_drops_device_sources_before_the_window()
    -> Result<(), Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![0.0],
            vec![true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let land = LandModel::impervious(&domain);
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: 0.02,
                qx: 0.0,
                qy: 0.0,
            }],
        )?;
        let forcing = ForcingSchedule::new(&domain, &boundaries, intervals(3))?;
        let mut solver = Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new()),
            SolverSetup {
                domain: domain.clone(),
                land,
                boundaries: boundaries.clone(),
                forcing,
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?;
        solver.advance_to(3.0, 1000)?;
        assert_eq!(solver.context.sources.len(), 3);
        let before: Option<StepDiagnostics> = solver.last_step()?;
        let tail = ForcingSchedule::new(&domain, &boundaries, intervals(4)[3..].to_vec())?;
        solver.continue_with_forcing(tail)?;
        assert_eq!(solver.context.sources.len(), 2);
        assert_eq!(solver.definition.prefix.intervals, 2);
        assert_eq!(solver.definition.chain.len(), 2);
        let Some(StepReceipt::Device(receipt)) = &solver.accepted.last else {
            return Err("accepted step must retain device receipts".into());
        };
        assert_eq!(receipt.forcing_index, 0);
        assert_eq!(solver.last_step()?, before);
        let whole = forcing_history::chain(&forcing_history::origin(), &intervals(4));
        assert_eq!(solver.definition.chain, whole[2..]);
        Ok(())
    }
}
