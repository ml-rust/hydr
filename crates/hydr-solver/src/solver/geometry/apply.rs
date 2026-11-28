//! Applying surface bed epochs while keeping depth, momenta, and storage bit-identical.

use crate::contract::EngineError;
use crate::model::Domain;
use crate::solver::compatibility::Candidate;
use crate::solver::coupling::SurfacePorts;
use crate::solver::definition::Definition;
use crate::solver::receipts::{StepReceipt, rebase_forcing_index};
use crate::solver::snapshot::Exchange;
use crate::solver::tensor::TensorDomain;
use crate::solver::{CONTRACT_VERSION, Snapshot, SnapshotData, Solver};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, ReduceOps, ScalarOps, ShapeOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedNewtonAlgorithms};
use std::sync::Arc;

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
        + CumulativeOps<R>
        + ShapeOps<R>
        + BatchedBisectionAlgorithms<R>
        + BatchedNewtonAlgorithms<R>,
{
    /// Returns a copy of this engine with updated cell `bed` elevations, keeping depth `h`,
    /// momenta `qx`/`qy`, soil state, and water ledger bit-identical.
    ///
    /// Cell area is unchanged, so stored water volume is conserved bit-for-bit while stage
    /// `bed + h` shifts by the bed change.
    #[doc(hidden)]
    pub fn coupling_with_bed(&self, bed: Vec<f64>) -> Result<Self, EngineError> {
        let old = &self.definition.domain;
        let domain = Domain::new(
            old.grid(),
            bed,
            old.manning().to_vec(),
            old.active().to_vec(),
            old.reference().clone(),
        )?;
        let definition = Arc::new(Definition::new(
            domain,
            self.definition.land.clone(),
            self.definition.boundaries.clone(),
            self.definition.forcing.clone(),
            self.definition.config.clone(),
            self.definition.prefix.clone(),
        )?);
        let context = Arc::new(TensorDomain::<R>::new(
            self.context.client.clone(),
            &definition.domain,
            &definition.forcing,
        )?);
        Ok(Self {
            definition,
            accepted: self.accepted.clone(),
            context,
            coastal: self.coastal,
            subgrid: self.subgrid.clone(),
        })
    }

    /// Restores a trusted coordinator snapshot whose surface bed may reflect a scheduled bed epoch.
    #[doc(hidden)]
    pub fn coupling_restore_with_bed(&mut self, snapshot: &Snapshot<R>) -> Result<(), EngineError> {
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
                .equivalent_ignoring_bed(&Candidate::of(source, retained))?
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
        self.definition = Arc::clone(&snapshot.definition);
        self.context = Arc::clone(&snapshot.context);
        self.accepted = accepted;
        Ok(())
    }

    /// Imports a coordinator checkpoint package whose surface bed may reflect a scheduled bed epoch.
    #[doc(hidden)]
    pub fn coupling_restore_data_with_bed(
        &mut self,
        data: &SnapshotData,
        ports: Option<&SurfacePorts<R>>,
    ) -> Result<(), EngineError> {
        self.import_data_with_bed(data, Exchange::Admit, ports, true)
    }
}
