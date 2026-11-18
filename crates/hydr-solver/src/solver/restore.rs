//! Trusted in-memory restoration from a same-device snapshot.

use super::compatibility::Candidate;
use super::receipts::{StepReceipt, rebase_forcing_index};
use super::snapshot::reject_exchange;
use super::{CONTRACT_VERSION, Snapshot, Solver};
use crate::contract::EngineError;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, ReduceOps,
    ScalarOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};

impl<R: Runtime<DType = DType>> Solver<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
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
    /// Restores complete accepted state after checking the immutable numerical definition.
    ///
    /// The snapshot and this engine can hold different forcing windows of one schedule.
    /// Both histories fold to the later prefix, then the folded prefixes and remaining intervals compare bitwise.
    /// This engine's window must hold the interval of the snapshot's last step.
    ///
    /// A snapshot with coordinator exchange water restores only through its coordinator.
    ///
    /// # Errors
    /// Rejects different devices, geometry, parameters, boundaries, forcing, contracts, an uncovered
    /// window, or coordinator exchange before mutation.
    pub fn restore(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        // Cumulative exchange covers the last step, so the ledger alone decides.
        reject_exchange(&snapshot.accepted.ledger, None)?;
        self.restore_snapshot(snapshot)
    }

    /// Restores a snapshot that can carry coordinator exchange water.
    ///
    /// Runs every [`Solver::restore`] check except the standalone exchange rejection.
    ///
    /// # Errors
    /// Rejects different devices, geometry, parameters, boundaries, forcing, contracts, or an uncovered window before mutation.
    #[doc(hidden)]
    pub fn coupling_restore(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        self.restore_snapshot(snapshot)
    }

    fn restore_snapshot(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
        let source = &snapshot.definition;
        let retained = source.retained_from(
            snapshot.accepted.time,
            snapshot.accepted.last.as_ref().map(StepReceipt::start),
        );
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.context.client.device())
            || snapshot.contract != CONTRACT_VERSION
            || !self
                .definition
                .equivalent(&Candidate::of(source, retained))?
        {
            return Err(EngineError::rejected(
                "snapshot",
                "definition or contract mismatch: restore into an engine built from the same definition and forcing schedule",
            ));
        }
        let mut land = self.accepted.land.clone();
        land.restore_trusted(&snapshot.accepted.land)?;
        let mut accepted = snapshot.accepted.clone();
        accepted.land = land;
        if let Some(StepReceipt::Device(receipt)) = &mut accepted.last {
            receipt.forcing_index = rebase_forcing_index(
                receipt.forcing_index,
                source.prefix.intervals,
                self.definition.prefix.intervals,
                self.context.sources.len(),
            )?;
            receipt.sources.soil.next = accepted.land.clone();
        }
        self.accepted = accepted;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::CartesianGrid;
    use crate::model::{CellState, Domain, FlowState, SpatialReference};
    use crate::soil::{LandCell, LandModel, ModifiedGreenAmptParameters, SoilState};
    use crate::solver::forcing_history::ForcingPrefix;
    use crate::solver::{
        BoundaryConditions, ForcingSchedule, InitialState, SolverConfig, SolverSetup,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    use std::sync::Arc;

    fn engine(client: CpuClient) -> Result<Solver<CpuRuntime>, Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![0.0],
            vec![true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let forcing = ForcingSchedule::new(
            &domain,
            &boundaries,
            vec![crate::contract::ForcingInterval {
                start: 0.0,
                end: 1.0,
                rain: vec![1e-6],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            }],
        )?;
        let parameters = ModifiedGreenAmptParameters::from_native(0.5, 1e-5, 0.3)?;
        let land = LandModel::new(&domain, vec![LandCell::ModifiedGreenAmpt(parameters)])?;
        let soil = SoilState::initialized(&land)?;
        let flow = FlowState::new(
            &domain,
            vec![CellState {
                h: 0.02,
                qx: 0.0,
                qy: 0.0,
            }],
        )?;
        Ok(Solver::<CpuRuntime>::new(
            client,
            SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            InitialState {
                time: 0.0,
                flow,
                soil,
            },
        )?)
    }

    #[test]
    fn trusted_restore_retains_destination_client_with_soil_and_receipts()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let source_config = ParallelismConfig::new(Some(1), Some(3));
        let destination_config = ParallelismConfig::new(Some(1), Some(7));
        let mut source = engine(CpuClient::new(device.clone()).with_parallelism(source_config))?;
        let mut destination = engine(CpuClient::new(device).with_parallelism(destination_config))?;
        source.advance_to(0.01, 100)?;
        destination.advance_to(0.02, 100)?;
        let snapshot = source.snapshot();
        let context = Arc::clone(&destination.context);
        destination.restore(&snapshot)?;
        assert!(Arc::ptr_eq(&context, &destination.context));
        assert_eq!(source.context.client.parallelism(), source_config);
        assert_eq!(destination.context.client.parallelism(), destination_config);
        assert_eq!(destination.snapshot().data()?, snapshot.data()?);
        let Some(super::super::receipts::StepReceipt::Device(receipt)) = &destination.accepted.last
        else {
            return Err("accepted step must retain device receipts".into());
        };
        assert_eq!(
            receipt.sources.soil.next.soil_state()?,
            destination.soil_state()?
        );
        source.advance_to(0.03, 100)?;
        destination.advance_to(0.03, 100)?;
        assert_eq!(destination.snapshot().data()?, source.snapshot().data()?);
        assert_eq!(destination.context.client.parallelism(), destination_config);
        Ok(())
    }

    /// Builds a three-interval engine whose window starts at interval `from`.
    fn windowed(
        client: CpuClient,
        from: usize,
    ) -> Result<Solver<CpuRuntime>, Box<dyn std::error::Error>> {
        let domain = Domain::new(
            CartesianGrid::new(1, 1, 1.0, 1.0)?,
            vec![0.0],
            vec![0.0],
            vec![true],
            SpatialReference::new("projected".into(), "datum".into(), [0.0; 2])?,
        )?;
        let boundaries = BoundaryConditions::impermeable_walls();
        let intervals: Vec<_> = (0..3)
            .map(|k| crate::contract::ForcingInterval {
                start: k as f64,
                end: (k + 1) as f64,
                rain: vec![1e-6 * (k + 1) as f64],
                potential_evaporation: vec![0.0],
                withdrawal: vec![0.0],
                boundaries: Vec::new(),
            })
            .collect();
        let prefix = super::super::forcing_history::folded(
            &ForcingPrefix::empty(0.0),
            &intervals,
            None,
            from,
        )
        .ok_or("prefix past the schedule")?;
        let forcing = ForcingSchedule::new(&domain, &boundaries, intervals[from..].to_vec())?;
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
        Ok(Solver::<CpuRuntime>::with_prefix(
            client,
            SolverSetup {
                domain,
                land,
                boundaries,
                forcing,
                config: SolverConfig::default(),
            },
            Some(prefix),
            InitialState {
                time: from as f64,
                flow,
                soil,
            },
        )?)
    }

    #[test]
    fn prefixed_window_restores_into_full_engine_with_rebased_receipt()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let client = || CpuClient::new(device.clone());
        let mut source = windowed(client(), 1)?;
        assert_eq!(source.context.sources.len(), 2);
        source.advance_to(1.5, 100)?;
        let mut trusted = windowed(client(), 0)?;
        let mut imported = windowed(client(), 0)?;
        trusted.restore(&source.snapshot())?;
        imported.restore_data(&source.snapshot().data()?)?;
        let Some(super::super::receipts::StepReceipt::Device(receipt)) = &trusted.accepted.last
        else {
            return Err("accepted step must retain device receipts".into());
        };
        assert_eq!(receipt.forcing_index, 1);
        assert_eq!(trusted.snapshot().data()?, source.snapshot().data()?);
        assert_eq!(trusted.last_step()?, source.last_step()?);
        for engine in [&mut source, &mut trusted, &mut imported] {
            engine.advance_to(3.0, 100)?;
        }
        assert_eq!(trusted.snapshot().data()?, source.snapshot().data()?);
        assert_eq!(imported.snapshot().data()?, source.snapshot().data()?);
        Ok(())
    }

    #[test]
    fn window_starting_after_checkpoint_interval_rejects_both_paths()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let source = windowed(CpuClient::new(device.clone()), 0)?;
        let mut late = windowed(CpuClient::new(device), 1)?;
        let before = late.snapshot().data()?;
        let error = late
            .restore(&source.snapshot())
            .err()
            .ok_or("restore must reject")?;
        assert!(
            error.to_string().contains("starts at interval 1"),
            "{error}"
        );
        assert_eq!(late.snapshot().data()?, before);
        let error = late
            .restore_data(&source.snapshot().data()?)
            .err()
            .ok_or("restore_data must reject")?;
        assert!(
            error.to_string().contains("starts at interval 1"),
            "{error}"
        );
        assert_eq!(late.snapshot().data()?, before);
        Ok(())
    }
}
