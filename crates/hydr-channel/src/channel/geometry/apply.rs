//! Applying channel section epochs while keeping wet area, discharge, and volume bit-identical.

use crate::channel::flow::context::Context;
use crate::channel::{
    ChannelError, ChannelGeometry, ChannelReach, ChannelReachDefinition, ChannelSnapshot,
    ChannelSnapshotData,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, HypotOps, IndexingOps, MidpointOps,
    NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::{Device, Runtime, RuntimeClient};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{
            BatchedBisectionAlgorithms, CertifiedBisectionAlgorithms,
            CertifiedUpperBracketAlgorithms,
        },
    },
};
use std::sync::Arc;

impl ChannelReachDefinition {
    /// True when two definitions match bit for bit in datum, provenance, lateral bindings, and
    /// config, ignoring cross-section geometry.
    #[doc(hidden)]
    pub fn same_bits_ignoring_geometry(&self, other: &Self) -> bool {
        let mut probe = other.clone();
        probe.geometry = self.geometry.clone();
        self.same_bits(&probe)
    }
}

impl<R> ChannelReach<R>
where
    R: Runtime<DType = DType>,
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + HypotOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TensorOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    /// Returns a copy of this reach with updated cross-section `geometry`, keeping cell volumes,
    /// longitudinal momenta, lateral counters, and water ledger bit-identical.
    ///
    /// Runs certified area-to-stage recovery on the new section; rejects if any cell's wet area
    /// exceeds the new section's bank limit.
    #[doc(hidden)]
    pub fn coupling_with_geometry(&self, geometry: ChannelGeometry) -> Result<Self, ChannelError> {
        let mut definition = self.definition().clone();
        definition.geometry = geometry;
        let context = Arc::new(Context::with_binding_owner(
            self.context.client.clone(),
            definition,
            self.context.original_initial.clone(),
            true,
        )?);
        // Check that current wet area fits within the new section's bank limit.
        context.fields(&self.accepted.volume, &self.accepted.momentum)?;
        Ok(Self {
            context,
            accepted: self.accepted.clone(),
        })
    }

    /// Restores a trusted coordinator snapshot whose reach geometry may reflect a scheduled
    /// section epoch.
    #[doc(hidden)]
    pub fn coupling_restore_with_geometry(
        &self,
        snapshot: &ChannelSnapshot<R>,
    ) -> Result<Self, ChannelError> {
        if !self
            .definition()
            .same_bits_ignoring_geometry(snapshot.reach.definition())
        {
            return Err(ChannelError::input(
                "channel checkpoint metadata or config bits differ from the destination",
            ));
        }
        if self.context.original_initial != snapshot.reach.context.original_initial {
            return Err(ChannelError::input(
                "channel checkpoint original initial-state bits differ from the destination",
            ));
        }
        if !self
            .context
            .client
            .device()
            .is_same(snapshot.reach.context.client.device())
        {
            return Err(ChannelError::input(
                "channel checkpoint trusted tensor device differs from the destination client",
            ));
        }
        let context = if self.definition().same_bits(snapshot.reach.definition()) {
            Arc::clone(&self.context)
        } else {
            Arc::new(Context::with_binding_owner(
                self.context.client.clone(),
                snapshot.reach.definition().clone(),
                self.context.original_initial.clone(),
                true,
            )?)
        };
        Ok(Self {
            context,
            accepted: snapshot.reach.accepted.clone(),
        })
    }

    /// Imports a coordinator channel package whose geometry may reflect a scheduled section epoch.
    #[doc(hidden)]
    pub fn coupling_import_snapshot_with_geometry(
        &self,
        data: &ChannelSnapshotData,
    ) -> Result<Self, ChannelError>
    where
        R::Client: CorrectlyRoundedOps<R>
            + BatchedBisectionAlgorithms<R>
            + CertifiedSimpsonAlgorithms<R>
            + CertifiedBisectionAlgorithms<R>
            + CertifiedMappedBisectionAlgorithms<R>
            + CertifiedUpperBracketAlgorithms<R>,
    {
        if !self
            .definition()
            .same_bits_ignoring_geometry(&data.definition)
        {
            return Err(ChannelError::input(
                "channel checkpoint metadata or config bits differ from the destination",
            ));
        }
        if self.definition().same_bits(&data.definition) {
            return self.coupling_import_snapshot(data);
        }
        let context = Arc::new(Context::with_binding_owner(
            self.context.client.clone(),
            data.definition.clone(),
            self.context.original_initial.clone(),
            true,
        )?);
        let base = Self {
            context,
            accepted: self.accepted.clone(),
        };
        base.coupling_import_snapshot(data)
    }
}
