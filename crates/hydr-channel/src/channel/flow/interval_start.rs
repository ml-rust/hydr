//! Accepted-start donor water, fields, face rates, and limits shared by stepping and coupling.

use super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, conserved_point::ConservedPoint,
    context::Context, flux::Flux,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ReduceOps, ScalarOps,
    ShapeOps, TensorOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedMappedBisectionAlgorithms,
        roots::{CertifiedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
    },
};

/// Values that depend only on the accepted state and the start endpoint policies.
pub(super) struct IntervalStart<R: Runtime<DType = DType>> {
    /// Water each cell can donate, in m³.
    pub donor: Tensor<R>,
    pub fields: Fields<R>,
    pub flux: Flux<R>,
    /// Native shared-face outflow from each cell, in m³/s.
    pub outgoing: Tensor<R>,
    /// Accepted-start CFL limit, in seconds.
    pub start_cfl: f64,
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: TensorOps<R>
        + NextRepresentableOps<R>
        + MidpointOps<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub fn interval_start(
        &self,
        volume: &ConservedPoint<R>,
        momentum: &ConservedPoint<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<IntervalStart<R>, ChannelError> {
        let donor = self.donor_volume(volume)?;
        let fields = self.fields(volume, momentum)?;
        let flux = self.flux(&fields, left, right)?;
        let start_cfl = self.cfl_limit(
            &flux.speed,
            flux.internal_speed.as_ref(),
            flux.dry_fronts.as_ref(),
        )?;
        let outgoing = self.outgoing_rate(&flux.water)?;
        Ok(IntervalStart {
            donor,
            fields,
            flux,
            outgoing,
            start_cfl,
        })
    }
}
