//! Retry loop that turns prepared and completed trials into the next accepted state.

use super::super::Solver;
use super::super::engine::Accepted;
use super::complete::Completion;
use super::prepare::{IntervalTrial, Preparation};
use crate::contract::{EngineError, ForcingInterval};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, HypotOps, IndexingOps, ReduceOps, ScalarOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

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
    /// Returns the index and forcing interval that cover the accepted time.
    ///
    /// # Errors
    /// Rejects an accepted time past the last explicit interval.
    pub(in crate::solver) fn forcing_interval(
        &self,
    ) -> Result<(usize, &ForcingInterval), EngineError> {
        let time = self.time();
        let index = self
            .definition
            .forcing
            .intervals()
            .partition_point(|interval| interval.end <= time);
        let forcing = self
            .definition
            .forcing
            .intervals()
            .get(index)
            .ok_or_else(|| {
                EngineError::rejected(
                    "forcing",
                    format!("time={time}: supply the next explicit interval"),
                )
            })?;
        Ok((index, forcing))
    }

    pub(in crate::solver) fn next_accepted(&self, target: f64) -> Result<Accepted<R>, EngineError> {
        let config = self.config();
        let time = self.time();
        let (index, forcing) = self.forcing_interval()?;
        let endpoint = target.min(forcing.end).min(time + config.max_step);
        let mut dt = endpoint - time;
        let mut last_reason = String::new();
        let mut last_source_error = None;
        for retry in 0..=config.max_retries {
            let end = if retry == 0 { endpoint } else { time + dt };
            dt = end - time;
            if !dt.is_finite() || dt <= 0.0 || (dt < config.min_step && end != endpoint) {
                if let Some(error) = last_source_error {
                    return Err(error);
                }
                return Err(EngineError::rejected(
                    "timestep",
                    format!(
                        "time={time}, dt={dt}, retry={retry}: cannot advance within min_step={}; inspect inputs and prior trial: {last_reason}",
                        config.min_step
                    ),
                ));
            }
            let trial = IntervalTrial {
                index,
                end,
                dt,
                retry,
            };
            let prepared = match self.prepare_interval(trial)? {
                Preparation::Ready(prepared) => {
                    last_source_error = None;
                    prepared
                }
                Preparation::SourceRetry(error) => {
                    last_reason = error.to_string();
                    last_source_error = Some(error);
                    dt *= 0.5;
                    continue;
                }
                Preparation::BoundaryRetry(error) => {
                    last_source_error = None;
                    last_reason = error.to_string();
                    dt *= 0.5;
                    continue;
                }
            };
            let limit = prepared.limit();
            if dt > limit {
                last_reason = format!(
                    "CFL limit={}, depletion limit={}",
                    prepared.cfl_limit, prepared.depletion_limit
                );
                // Room for time subtraction avoids repeating an identical rounded trial.
                dt = 0.95 * limit;
                continue;
            }
            match self.complete_interval(*prepared, None)? {
                Completion::Accepted(accepted) => return Ok(*accepted),
                Completion::Retry(reason) => {
                    last_reason = reason;
                    dt *= 0.5;
                }
            }
        }
        if let Some(error) = last_source_error {
            return Err(error);
        }
        Err(EngineError::rejected(
            "trial retries",
            format!("time={time}, limit={}: {last_reason}", config.max_retries),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::{
        Axis, BoundaryConditions, BoundaryFace, BoundaryForcing, ForcingSchedule, SolverConfig,
        StageBoundary,
    };
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandCell, LandModel, ModifiedGreenAmptParameters, SoilState};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult<T> = Result<T, Box<dyn std::error::Error>>;

    /// Two cells on a slope with rain, Green-Ampt soil, and an inflowing stage face.
    fn engine() -> TestResult<Solver<CpuRuntime>> {
        let domain = Domain::new(
            CartesianGrid::new(2, 1, 0.1, 0.1)?,
            vec![0.0, 0.05],
            vec![0.03, 0.03],
            vec![true, true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let face = BoundaryFace {
            axis: Axis::X,
            column: 0,
            row: 0,
        };
        let boundaries = BoundaryConditions::with_open_faces(&domain, vec![face])?;
        let intervals = [(0.0, 0.05, 0.3), (0.05, 0.1, 0.25)]
            .into_iter()
            .map(|(start, end, elevation)| ForcingInterval {
                start,
                end,
                rain: vec![1e-4, 2e-4],
                potential_evaporation: vec![0.0, 0.0],
                withdrawal: vec![0.0, 0.0],
                boundaries: vec![BoundaryForcing::Stage(StageBoundary {
                    face,
                    elevation,
                    inflow_tangential_velocity: Some(0.0),
                })],
            })
            .collect();
        let forcing = ForcingSchedule::new(&domain, &boundaries, intervals)?;
        let parameters = ModifiedGreenAmptParameters::from_native(0.5, 1e-5, 0.3)?;
        let land = LandModel::new(&domain, vec![LandCell::ModifiedGreenAmpt(parameters); 2])?;
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(
            &domain,
            vec![
                CellState {
                    h: 0.2,
                    qx: 0.0,
                    qy: 0.0,
                },
                CellState {
                    h: 0.15,
                    qx: -0.0,
                    qy: 0.0,
                },
            ],
        )?;
        Ok(Solver::<CpuRuntime>::new(
            CpuClient::new(CpuDevice::new())
                .with_parallelism(ParallelismConfig::new(Some(1), None)),
            crate::solver::SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            crate::solver::InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?)
    }

    #[test]
    fn uncoupled_split_step_is_bit_identical() -> TestResult<()> {
        let mut reference = engine()?;
        let mut plan = Vec::new();
        reference.advance_controlled(
            0.1,
            10_000,
            || false,
            |step, _| plan.push((step.start, step.end, step.dt, step.retries)),
        )?;
        assert!(plan.len() > 2, "fixture must accept several steps");
        assert!(
            plan.iter().any(|step| step.3 > 0),
            "fixture must exercise the retry loop"
        );
        let mut split = engine()?;
        for (start, end, dt, retry) in plan {
            assert_eq!(split.time().to_bits(), start.to_bits());
            let (index, _) = split.forcing_interval()?;
            let trial = IntervalTrial {
                index,
                end,
                dt,
                retry,
            };
            let Preparation::Ready(prepared) = split.prepare_interval(trial)? else {
                return Err("an accepted step must prepare without retry".into());
            };
            assert!(dt <= prepared.limit());
            let Completion::Accepted(accepted) = split.complete_interval(*prepared, None)? else {
                return Err("an accepted step must complete without retry".into());
            };
            split.accepted = *accepted;
        }
        let expected = reference.snapshot().data()?;
        let actual = split.snapshot().data()?;
        // Debug output spells signed zeros and shortest round-trip values, so equal text is equal bits.
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(
            actual.accepted.time.to_bits(),
            expected.accepted.time.to_bits()
        );
        let bits = |data: &crate::solver::SnapshotData| -> Vec<u64> {
            data.accepted
                .flow
                .cells()
                .iter()
                .flat_map(|cell| [cell.h, cell.qx, cell.qy])
                .map(f64::to_bits)
                .collect()
        };
        assert_eq!(bits(&actual), bits(&expected));
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    fn coupling_complete_none_matches_standalone_trial() -> TestResult<()> {
        use crate::solver::coupling::{CouplingCompletion, CouplingPreparation};
        let mut reference = engine()?;
        let mut plan = Vec::new();
        reference.advance_controlled(
            0.1,
            10_000,
            || false,
            |step, _| plan.push((step.start, step.end, step.retries)),
        )?;
        assert!(plan.iter().any(|step| step.2 > 0));
        let mut coupled = engine()?;
        let links = coupled.coupling_links(&[1])?;
        for (start, end, retry) in plan {
            assert_eq!(coupled.time().to_bits(), start.to_bits());
            let CouplingPreparation::Ready(stage) = coupled.coupling_prepare(end, retry, &links)?
            else {
                return Err("an accepted step must prepare without retry".into());
            };
            assert!(stage.dt() <= stage.limit());
            let before = coupled.snapshot().data()?;
            let CouplingCompletion::Accepted(next) = coupled.coupling_complete(&stage, None)?
            else {
                return Err("an accepted step must complete without retry".into());
            };
            // The calling engine keeps its accepted state.
            assert_eq!(
                format!("{:?}", coupled.snapshot().data()?),
                format!("{before:?}")
            );
            coupled = *next;
        }
        let expected = reference.snapshot().data()?;
        let actual = coupled.snapshot().data()?;
        assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        assert_eq!(actual, expected);
        let ledger = coupled.ledger();
        assert_eq!(ledger.exchange_in_m3.to_bits(), 0.0_f64.to_bits());
        assert_eq!(ledger.exchange_out_m3.to_bits(), 0.0_f64.to_bits());
        let step = coupled
            .last_step()?
            .ok_or("coupled run must record a step")?;
        for value in [
            step.exchange_in_volume,
            step.exchange_out_volume,
            step.exchange_representation_error,
        ] {
            assert_eq!(value.to_bits(), 0.0_f64.to_bits());
        }
        Ok(())
    }
}
