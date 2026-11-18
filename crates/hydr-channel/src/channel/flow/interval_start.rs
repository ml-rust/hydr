//! Accepted-start donor water, fields, face rates, and limits shared by stepping and coupling.

use super::{
    ChannelError, ChannelTensorBoundary, admission::Fields, conserved_point::ConservedPoint,
    context::Context, engine::Accepted, flux::Flux,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, MidpointOps, NextRepresentableOps,
    ReduceOps, ScalarOps, ShapeOps, TensorOps, TwoSumOps, UnaryOps,
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
use std::ops::Deref;

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

/// An [`IntervalStart`] kept with the accepted state and endpoint policies it came from.
///
/// A coordinator calculates it once per common trial and reuses it for every candidate.
pub(super) struct ReservedStart<R: Runtime<DType = DType>> {
    pub start: IntervalStart<R>,
    pub left: ChannelTensorBoundary<R>,
    pub right: ChannelTensorBoundary<R>,
    time_bits: u64,
    steps: usize,
    volume: ConservedPoint<R>,
    momentum: ConservedPoint<R>,
}

impl<R: Runtime<DType = DType>> ReservedStart<R> {
    /// Keeps `start` with the `accepted` state and the endpoints it was calculated from.
    pub fn new(
        start: IntervalStart<R>,
        accepted: &Accepted<R>,
        left: ChannelTensorBoundary<R>,
        right: ChannelTensorBoundary<R>,
    ) -> Self {
        Self {
            start,
            left,
            right,
            time_bits: accepted.time.to_bits(),
            steps: accepted.steps,
            volume: accepted.volume.clone(),
            momentum: accepted.momentum.clone(),
        }
    }

    /// True when `accepted` holds the same time, step count, and conserved storage.
    pub fn belongs_to(&self, accepted: &Accepted<R>) -> bool {
        let same = |a: &ConservedPoint<R>, b: &ConservedPoint<R>| {
            a.high.shares_storage_with(&b.high) && a.low.shares_storage_with(&b.low)
        };
        self.time_bits == accepted.time.to_bits()
            && self.steps == accepted.steps
            && same(&self.volume, &accepted.volume)
            && same(&self.momentum, &accepted.momentum)
    }
}

impl<R: Runtime<DType = DType>> std::fmt::Debug for ReservedStart<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReservedStart")
            .field("start_cfl", &self.start.start_cfl)
            .field("time", &f64::from_bits(self.time_bits))
            .field("steps", &self.steps)
            .finish_non_exhaustive()
    }
}

/// A borrowed reserved start or one calculated for this step.
pub(super) enum StartRef<'a, R: Runtime<DType = DType>> {
    Reserved(&'a IntervalStart<R>),
    Calculated(Box<IntervalStart<R>>),
}

impl<R: Runtime<DType = DType>> Deref for StartRef<'_, R> {
    type Target = IntervalStart<R>;

    fn deref(&self) -> &IntervalStart<R> {
        match self {
            Self::Reserved(start) => start,
            Self::Calculated(start) => start,
        }
    }
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
        + CorrectlyRoundedOps<R>
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

    /// The `reserved` start when it belongs to `accepted`, else a fresh calculation.
    ///
    /// Both paths run the same calculation on the same operands, so every bit agrees.
    pub fn select_start<'a>(
        &self,
        reserved: Option<&'a ReservedStart<R>>,
        accepted: &Accepted<R>,
        left: &ChannelTensorBoundary<R>,
        right: &ChannelTensorBoundary<R>,
    ) -> Result<StartRef<'a, R>, ChannelError> {
        match reserved.filter(|reserved| reserved.belongs_to(accepted)) {
            Some(reserved) => Ok(StartRef::Reserved(&reserved.start)),
            None => Ok(StartRef::Calculated(Box::new(self.interval_start(
                &accepted.volume,
                &accepted.momentum,
                left,
                right,
            )?))),
        }
    }
}
