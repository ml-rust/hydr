//! Host projections of structure links, pool ends, and pools: interval receipts and lifetime
//! ledgers.

use super::batch::SurfaceChannelBatch;
use super::context::Context;
use super::pool_ends::{PoolEndCounters, PoolEndInterval};
use super::pools::{PoolChild, inflow_error, pool_error};
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelError, SurfaceChannelOperationReceipt, SurfaceChannelPoolEndLedger,
    SurfaceChannelPoolEndTransfer, SurfaceChannelPoolStep, SurfaceChannelStructureLedger,
    SurfaceChannelStructureTransfer,
};
use crate::coupling::level_pools::CoupledPoolInflowLedger;
use crate::structures::{OperatingSourceIntervalDetail, StructureRegime};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use std::collections::BTreeMap;

/// Per-link host values of one accepted interval, each in link table order.
pub(super) struct LinkValues {
    pub signed: Vec<f64>,
    pub law: Vec<f64>,
    pub level: Vec<f64>,
    pub factor: Vec<f64>,
    pub regime: Vec<f64>,
    pub midpoint_a: Vec<f64>,
    pub midpoint_b: Vec<f64>,
}

impl<R: Runtime<DType = DType>> Context<R> {
    /// Link table positions of the structure links, ascending: every position outside `bank`.
    pub(super) fn structure_positions(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.link_count).filter(|index| self.bank.binary_search(index).is_err())
    }

    /// Pool-end identities in ascending order. Empty without pool ends.
    pub(super) fn pool_end_ids(&self) -> Vec<&str> {
        self.pool_ends.as_ref().map_or_else(Vec::new, |ends| {
            ends.ends.iter().map(|end| end.id.as_str()).collect()
        })
    }
}

impl<R: Runtime<DType = DType>> StoredStep<R> {
    /// Host receipts of every structure link of `context`, in ascending identity.
    pub(super) fn structure_transfers(
        &self,
        context: &Context<R>,
        values: &LinkValues,
    ) -> Result<Vec<SurfaceChannelStructureTransfer>, SurfaceChannelError> {
        let mut receipts = BTreeMap::new();
        if let Some(operation) = &self.operation {
            for receipt in operation.interval.receipts().map_err(|error| {
                SurfaceChannelError::Numerical(format!("operating source receipts: {error}"))
            })? {
                receipts.insert(receipt.structure_id, receipt.detail);
            }
        }
        let points = match &self.operation {
            Some(operation) => &operation.points,
            None => &context.operations.points,
        };
        context
            .structure_positions()
            .map(|index| {
                let id = &context.link_ids[index];
                let missing = || {
                    SurfaceChannelError::Numerical(format!(
                        "structure={id:?} lacks its interval receipt"
                    ))
                };
                Ok(SurfaceChannelStructureTransfer {
                    structure_id: id.clone(),
                    signed_volume_m3: *values.signed.get(index).ok_or_else(missing)?,
                    law_volume_m3: *values.law.get(index).ok_or_else(missing)?,
                    cap_factor: *values.level.get(index).ok_or_else(missing)?,
                    donor_factor: *values.factor.get(index).ok_or_else(missing)?,
                    regime: structure_regime(*values.regime.get(index).ok_or_else(missing)?)?,
                    midpoint_stage_a_m: *values.midpoint_a.get(index).ok_or_else(missing)?,
                    midpoint_stage_b_m: *values.midpoint_b.get(index).ok_or_else(missing)?,
                    operation: *points.get(index).ok_or_else(missing)?,
                    receipt: match receipts.remove(id) {
                        Some(detail) => receipt(detail),
                        None => SurfaceChannelOperationReceipt::Manual,
                    },
                })
            })
            .collect()
    }
}

/// Public receipt of one operating source detail.
fn receipt(detail: OperatingSourceIntervalDetail) -> SurfaceChannelOperationReceipt {
    match detail {
        OperatingSourceIntervalDetail::Manual => SurfaceChannelOperationReceipt::Manual,
        OperatingSourceIntervalDetail::PrescribedGate {
            start_opening_m,
            end_opening_m,
            opening_m,
            absolute_error_m,
            midpoint_rounding_m,
            closed,
        } => SurfaceChannelOperationReceipt::PrescribedGate {
            start_opening_m,
            end_opening_m,
            opening_m,
            absolute_error_m,
            midpoint_rounding_m,
            closed,
        },
        OperatingSourceIntervalDetail::ActuatedGate {
            actuator_id,
            start,
            end,
            opening_m,
            lower_opening_m,
            upper_opening_m,
            absolute_error_m,
            midpoint_rounding_m,
            closed,
        } => SurfaceChannelOperationReceipt::ActuatedGate {
            actuator_id,
            start,
            end,
            opening_m,
            lower_opening_m,
            upper_opening_m,
            absolute_error_m,
            midpoint_rounding_m,
            closed,
        },
        OperatingSourceIntervalDetail::ControlledPump {
            controller_id,
            latch,
            response_model,
            ..
        } => SurfaceChannelOperationReceipt::ControlledPump {
            controller_id,
            latch,
            response_model,
        },
    }
}

/// Law regime of a structures regime code.
fn structure_regime(code: f64) -> Result<StructureRegime, SurfaceChannelError> {
    [
        (1.0_f64, StructureRegime::SubmergedOrifice),
        (2.0, StructureRegime::FreeWeirOverflow),
        (3.0, StructureRegime::SubmergedWeirOverflow),
        (4.0, StructureRegime::BelowCrest),
        (5.0, StructureRegime::EqualHeads),
        (6.0, StructureRegime::PumpCurve),
        (7.0, StructureRegime::PumpIsolated),
        (8.0, StructureRegime::GateClosed),
        (9.0, StructureRegime::GateBelowSill),
        (10.0, StructureRegime::FreeGateAperture),
        (11.0, StructureRegime::MixedGateAperture),
        (12.0, StructureRegime::SubmergedGateAperture),
        (13.0, StructureRegime::CulvertInletControl),
        (14.0, StructureRegime::CulvertOutletControl),
        (15.0, StructureRegime::CulvertControlTie),
        (16.0, StructureRegime::CulvertBelowInvert),
        (17.0, StructureRegime::CulvertUndeclaredEnd),
        (18.0, StructureRegime::Closed),
    ]
    .into_iter()
    .find(|(value, _)| value.to_bits() == code.to_bits())
    .map(|(_, regime)| regime)
    .ok_or_else(|| {
        SurfaceChannelError::Numerical(format!("structure regime code={code} is no law regime"))
    })
}

/// Host receipts of every pool end of `ids`, in ascending identity.
pub(super) fn pool_end_transfers(
    ids: &[&str],
    interval: &PoolEndInterval,
) -> Result<Vec<SurfaceChannelPoolEndTransfer>, SurfaceChannelError> {
    if interval.volume_m3.len() != ids.len() || interval.head_m.len() != ids.len() {
        return Err(SurfaceChannelError::Numerical(
            "the pool-end receipt differs from the pool ends".into(),
        ));
    }
    Ok(ids
        .iter()
        .zip(interval.volume_m3.iter().zip(&interval.head_m))
        .map(|(id, (&volume, &head))| SurfaceChannelPoolEndTransfer {
            pool_end_id: (*id).to_string(),
            signed_volume_m3: volume,
            reservoir_head_m: head,
        })
        .collect())
}

impl PoolEndCounters {
    /// Host ledgers of the pool ends of `ids`, in ascending identity.
    pub(super) fn ledgers(&self, ids: &[&str]) -> Vec<SurfaceChannelPoolEndLedger> {
        ids.iter()
            .zip(&self.to_pool_m3)
            .zip(self.to_reach_m3.iter().zip(&self.roundoff_bound_m3))
            .map(
                |((id, &reach_to_pool_m3), (&pool_to_reach_m3, &roundoff_bound_m3))| {
                    SurfaceChannelPoolEndLedger {
                        pool_end_id: (*id).to_string(),
                        reach_to_pool_m3,
                        pool_to_reach_m3,
                        roundoff_bound_m3,
                    }
                },
            )
            .collect()
    }
}

impl<R: Runtime<DType = DType>> LinkCounters<R> {
    /// Host counters of the structure links of `context`, in ascending identity.
    pub(super) fn structure_ledgers(
        &self,
        context: &Context<R>,
    ) -> Result<Vec<SurfaceChannelStructureLedger>, SurfaceChannelError> {
        let a_to_b = self.to_channel_m3.try_to_vec::<f64>()?;
        let b_to_a = self.to_surface_m3.try_to_vec::<f64>()?;
        let bound = self.roundoff_bound_m3.try_to_vec::<f64>()?;
        Ok(context
            .structure_positions()
            .map(|index| SurfaceChannelStructureLedger {
                structure_id: context.link_ids[index].clone(),
                a_to_b_m3: a_to_b[index],
                b_to_a_m3: b_to_a[index],
                roundoff_bound_m3: bound[index],
            })
            .collect())
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    /// Lifetime external inflow ledgers of `pools`, in declared binding order. Empty without
    /// pools.
    pub(super) fn pool_inflow_ledgers(
        &self,
        pools: Option<&PoolChild<R>>,
    ) -> Result<Vec<CoupledPoolInflowLedger>, SurfaceChannelError> {
        match pools {
            Some(pools) => self
                .pool_topology()?
                .inflows
                .project_state(&self.client, &pools.external)
                .map_err(inflow_error),
            None => Ok(Vec::new()),
        }
    }
}

impl<R: Runtime<DType = DType>> Context<R> {
    /// Every pool of `pools` at its accepted end, in ascending identity. Empty without pools.
    pub(super) fn pool_steps(
        &self,
        pools: Option<&PoolChild<R>>,
    ) -> Result<Vec<SurfaceChannelPoolStep>, SurfaceChannelError> {
        let Some(pools) = pools else {
            return Ok(Vec::new());
        };
        let topology = self.pool_topology()?;
        let stage = (topology.fns.stages)(&pools.batch)
            .map_err(pool_error)?
            .try_to_vec::<f64>()?;
        let native = pools.batch.coupling_native_interval().ok_or_else(|| {
            SurfaceChannelError::Numerical("the accepted pools lack their interval".into())
        })?;
        let incoming = native.incoming_volume_m3.try_to_vec::<f64>()?;
        let outgoing = native.outgoing_volume_m3.try_to_vec::<f64>()?;
        let volume = native.volume_m3.try_to_vec::<f64>()?;
        let count = topology.ids.len();
        if [stage.len(), incoming.len(), outgoing.len(), volume.len()]
            .iter()
            .any(|&len| len != count)
        {
            return Err(SurfaceChannelError::Numerical(
                "the pool interval differs from the pools".into(),
            ));
        }
        Ok((0..count)
            .map(|index| SurfaceChannelPoolStep {
                pool_id: topology.ids[index].clone(),
                volume_m3: volume[index],
                stage_m: stage[index],
                incoming_m3: incoming[index],
                outgoing_m3: outgoing[index],
            })
            .collect())
    }
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    /// Projects lifetime directional structure counters in ascending structure identity.
    ///
    /// Empty without structure links.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn structure_ledgers(
        &self,
    ) -> Result<Vec<SurfaceChannelStructureLedger>, SurfaceChannelError> {
        self.counters.structure_ledgers(&self.context)
    }

    /// Projects lifetime directional pool-end counters in ascending pool-end identity.
    ///
    /// Empty without pool ends.
    ///
    /// # Errors
    /// Pool-end counters stay on the host, so no error occurs. The `Result` matches
    /// [`Self::structure_ledgers`].
    pub fn pool_end_ledgers(
        &self,
    ) -> Result<Vec<SurfaceChannelPoolEndLedger>, SurfaceChannelError> {
        Ok(self
            .counters
            .pool_ends
            .ledgers(&self.context.pool_end_ids()))
    }

    /// Projects lifetime external pool inflow ledgers in declared binding order.
    ///
    /// Empty without pool inflow bindings.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn pool_inflow_ledgers(&self) -> Result<Vec<CoupledPoolInflowLedger>, SurfaceChannelError> {
        self.context.pool_inflow_ledgers(self.pools.as_ref())
    }
}
