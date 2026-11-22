//! Host projections of the last interval, the link counters, and the composite balance.

use super::batch::SurfaceChannelBatch;
use super::context::Context;
use super::node_projection::{LinkValues, pool_end_transfers};
use super::pools::PoolChild;
use super::state::{LinkCounters, StoredStep};
use super::{
    SurfaceChannelError, SurfaceChannelLinkLedger, SurfaceChannelLinkTransfer,
    SurfaceChannelPortLedger, SurfaceChannelStepDiagnostics, SurfaceChannelWaterBalance,
    SurfaceChannelWeirRegime,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, HypotOps,
    IndexingOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps, TensorOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R> {
    /// Projects the last accepted common interval, absent before the first.
    ///
    /// # Errors
    /// Returns runtime transfer errors and [`SurfaceChannelError::Numerical`] for an unknown
    /// weir regime code.
    pub fn last_step(&self) -> Result<Option<SurfaceChannelStepDiagnostics>, SurfaceChannelError> {
        self.last
            .as_ref()
            .map(|step| step.diagnostics(&self.context, self.pools.as_ref()))
            .transpose()
    }

    /// Projects lifetime directional counters in ascending link identity.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn link_ledgers(&self) -> Result<Vec<SurfaceChannelLinkLedger>, SurfaceChannelError> {
        self.counters.ledgers(&self.context)
    }

    /// Projects lifetime directional port counters in ascending port identity.
    ///
    /// Each ledger counts the reach side's volume. Empty without ports.
    ///
    /// # Errors
    /// Port counters stay on the host, so no error occurs. The `Result` matches
    /// [`Self::link_ledgers`].
    pub fn port_ledgers(&self) -> Result<Vec<SurfaceChannelPortLedger>, SurfaceChannelError> {
        Ok(self.counters.ports.ledgers(self.context.port_ids()))
    }
}

impl<R: Runtime<DType = DType>> StoredStep<R> {
    /// Host receipt of this interval, with `pools` the accepted pools at its end.
    ///
    /// Bank links of `context` fill the per-link list and structure links the per-structure
    /// list, each in identity order.
    pub(super) fn diagnostics(
        &self,
        context: &Context<R>,
        pools: Option<&PoolChild<R>>,
    ) -> Result<SurfaceChannelStepDiagnostics, SurfaceChannelError> {
        let step = self;
        let values = LinkValues {
            signed: step.signed_m3.try_to_vec::<f64>()?,
            law: step.weir_m3.try_to_vec::<f64>()?,
            level: step.level.try_to_vec::<f64>()?,
            factor: step.factor.try_to_vec::<f64>()?,
            regime: step.regime.try_to_vec::<f64>()?,
            midpoint_a: step.midpoint_surface_m.try_to_vec::<f64>()?,
            midpoint_b: step.midpoint_channel_m.try_to_vec::<f64>()?,
        };
        let mut links = Vec::with_capacity(context.bank.len());
        for &index in &context.bank {
            links.push(SurfaceChannelLinkTransfer {
                link_id: context.link_ids[index].clone(),
                signed_volume_m3: values.signed[index],
                weir_volume_m3: values.law[index],
                level_factor: values.level[index],
                donor_factor: values.factor[index],
                regime: weir_regime(values.regime[index])?,
                midpoint_surface_stage_m: values.midpoint_a[index],
                midpoint_channel_stage_m: values.midpoint_b[index],
            });
        }
        let structures = step.structure_transfers(context, &values)?;
        let pool_ends = pool_end_transfers(&context.pool_end_ids(), &step.pool_ends)?;
        let totals = step.totals;
        Ok(SurfaceChannelStepDiagnostics {
            start_s: step.start_s,
            end_s: step.end_s,
            requested_end_s: step.requested_end_s,
            retries: step.retries,
            iterations: step.iterations,
            contraction: step.contraction,
            head_residual_m: step.head_residual_m,
            transfer_residual_m3: step.transfer_residual_m3,
            storage_before_m3: totals.storage_before_m3,
            storage_after_m3: totals.storage_after_m3,
            surface_exchange_in_m3: totals.surface_in_m3,
            surface_exchange_out_m3: totals.surface_out_m3,
            channel_exchange_in_m3: totals.channel_in_m3,
            channel_exchange_out_m3: totals.channel_out_m3,
            paired_bound_m3: totals.paired_bound_m3,
            mass_residual_m3: totals.mass_residual_m3,
            mass_bound_m3: totals.mass_bound_m3,
            links,
            ports: step.ports.transfers.clone(),
            port_paired_bound_m3: step.ports.paired_bound_m3,
            structures,
            pool_ends,
            pools: context.pool_steps(pools)?,
            runoff: step.runoff_transfers(context),
        })
    }
}

impl<R: Runtime<DType = DType>> LinkCounters<R> {
    /// Host counters of the bank links of `context` in identity order.
    pub(super) fn ledgers(
        &self,
        context: &Context<R>,
    ) -> Result<Vec<SurfaceChannelLinkLedger>, SurfaceChannelError> {
        let to_channel = self.to_channel_m3.try_to_vec::<f64>()?;
        let to_surface = self.to_surface_m3.try_to_vec::<f64>()?;
        let bound = self.roundoff_bound_m3.try_to_vec::<f64>()?;
        Ok(context
            .bank
            .iter()
            .map(|&index| SurfaceChannelLinkLedger {
                link_id: context.link_ids[index].clone(),
                surface_to_channel_m3: to_channel[index],
                channel_to_surface_m3: to_surface[index],
                roundoff_bound_m3: bound[index],
            })
            .collect())
    }
}

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
    /// Lifetime water balance of the surface, every reach, and the pools as one system.
    ///
    /// # Errors
    /// Returns runtime errors.
    pub fn ledger(&self) -> Result<SurfaceChannelWaterBalance, SurfaceChannelError> {
        self.context.balance(
            &self.definition.setup().config,
            (&self.surface, &self.reaches, self.pools.as_ref()),
            &self.counters,
            self.runoff.as_ref(),
        )
    }
}

/// Weir regime of a structures regime code.
fn weir_regime(code: f64) -> Result<SurfaceChannelWeirRegime, SurfaceChannelError> {
    [
        (2.0_f64, SurfaceChannelWeirRegime::Free),
        (3.0, SurfaceChannelWeirRegime::Submerged),
        (4.0, SurfaceChannelWeirRegime::BelowCrest),
        (5.0, SurfaceChannelWeirRegime::EqualHeads),
    ]
    .into_iter()
    .find(|(value, _)| value.to_bits() == code.to_bits())
    .map(|(_, regime)| regime)
    .ok_or_else(|| {
        SurfaceChannelError::Numerical(format!("weir regime code={code} is not a weir regime"))
    })
}
