use super::{
    root_invocation::{RootInvocation, RootInvocationWitness},
    wave_curve_types::CurveValue,
};
use crate::channel::{
    flow::surveyed::error::{HydraulicError, HydraulicResult},
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::{CertifiedExpansion, Interval},
    optimize::{CertifiedRootContinuation, CertifiedRootTolerance},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SonicSide {
    Left,
    Right,
}
impl SonicSide {
    pub(super) fn index(self) -> usize {
        match self {
            Self::Left => 0,
            Self::Right => 1,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i64)]
pub(super) enum SourceSonicStatus {
    NotRequested = 0,
    PositiveRoot = 1,
    DryContact = 2,
    SourceContact = 3,
    NoRootDry = 4,
    NoRootSource = 5,
    Unresolved = 6,
    DrySourceUnavailable = 7,
    WorkLimit = 8,
}

struct SonicRequestIdentity {
    invocation: RootInvocationWitness,
    side: SonicSide,
}
pub(super) struct SourceSonicBinding<R: Runtime<DType = DType>> {
    identity: Arc<SonicRequestIdentity>,
    pub(super) side: SonicSide,
    pub(super) owners: Tensor<R>,
    pub(super) rows: Tensor<R>,
}
impl<R: Runtime<DType = DType>> Clone for SourceSonicBinding<R> {
    fn clone(&self) -> Self {
        Self {
            identity: Arc::clone(&self.identity),
            side: self.side,
            owners: self.owners.clone(),
            rows: self.rows.clone(),
        }
    }
}
impl<R: Runtime<DType = DType>> SourceSonicBinding<R> {
    pub(super) fn new(
        invocation: &RootInvocation,
        side: SonicSide,
        owners: Tensor<R>,
        rows: Tensor<R>,
    ) -> Self {
        Self {
            identity: Arc::new(SonicRequestIdentity {
                invocation: invocation.witness(),
                side,
            }),
            side,
            owners,
            rows,
        }
    }
    pub(super) fn check(
        &self,
        invocation: &RootInvocation,
        side: SonicSide,
        nested: &Self,
    ) -> HydraulicResult<()> {
        invocation.check(&self.identity.invocation)?;
        invocation.check(&nested.identity.invocation)?;
        if self.side != self.identity.side
            || nested.side != nested.identity.side
            || side != self.side
            || side != nested.side
            || !Arc::ptr_eq(&self.identity, &nested.identity)
        {
            return Err(HydraulicError::Numerical { context: "Source sonic state belongs to another side or request. Preserve the original bound continuation".into() });
        }
        Ok(())
    }
    pub(super) fn selected(&self, owners: Tensor<R>, rows: Tensor<R>) -> Self {
        Self {
            identity: Arc::clone(&self.identity),
            side: self.side,
            owners,
            rows,
        }
    }
}

pub(super) struct SourceSonicContinuation<R: Runtime<DType = DType>> {
    pub(super) binding: SourceSonicBinding<R>,
    pub(super) continuation: CertifiedRootContinuation<R>,
    pub(super) chart: SupportedAreaQuery<R>,
}

pub(super) struct SourceSonicDomain<R: Runtime<DType = DType>> {
    pub(super) binding: SourceSonicBinding<R>,
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) epsilon: CertifiedExpansion<R>,
    pub(super) classification: SourceSonicStatus,
    pub(super) depth: Interval<R>,
    pub(super) normalized_depth: Interval<R>,
    pub(super) status: Tensor<R>,
    pub(super) iterations: usize,
    pub(super) callback_points: usize,
    pub(super) mapped_points: usize,
    pub(super) refinement: Option<SourceSonicContinuation<R>>,
}

pub(super) struct SourceSonicRequest<R: Runtime<DType = DType>> {
    pub(super) binding: SourceSonicBinding<R>,
    pub(super) status: Tensor<R>,
    pub(super) dry_residual: CurveValue<R>,
    pub(super) source_residual: CurveValue<R>,
    pub(super) domains: Vec<SourceSonicDomain<R>>,
    pub(super) tolerance: CertifiedRootTolerance,
    pub(super) terminal: bool,
    pub(super) endpoint_points: usize,
    pub(super) bracket_points: usize,
    pub(super) bracket_expansions: usize,
}

pub(super) struct SourceSonicState<R: Runtime<DType = DType>> {
    pub(super) status: [Tensor<R>; 2],
    pub(super) requests: Vec<SourceSonicRequest<R>>,
}
impl<R: Runtime<DType = DType>> SourceSonicState<R> {
    pub(super) fn empty(client: &R::Client, count: usize) -> HydraulicResult<Self> {
        Ok(Self {
            status: [
                Tensor::zeros(&[count], DType::I64, client.device())?,
                Tensor::zeros(&[count], DType::I64, client.device())?,
            ],
            requests: Vec::new(),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SonicFanMembership {
    NotChecked,
}

pub(super) struct SourceSonicCertificate<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) depth: Interval<R>,
    pub(super) normalized_depth: Interval<R>,
    pub(super) sound: Interval<R>,
    pub(super) velocity: Interval<R>,
    pub(super) discharge: Interval<R>,
    pub(super) source_discharge: CertifiedExpansion<R>,
    pub(super) area_difference: CertifiedExpansion<R>,
    pub(super) velocity_change: CertifiedExpansion<R>,
    pub(super) discharge_change: CertifiedExpansion<R>,
    pub(super) anchored_discharge: CertifiedExpansion<R>,
    pub(super) residual: Interval<R>,
    pub(super) classification: SourceSonicStatus,
    pub(super) status: Tensor<R>,
    pub(super) depth_status: Tensor<R>,
    pub(super) iterations: usize,
    pub(super) callback_points: usize,
    pub(super) mapped_points: usize,
    pub(super) fan_membership: SonicFanMembership,
}

pub(super) struct SourceSonicEvidence<R: Runtime<DType = DType>> {
    pub(super) invocation_owner: Tensor<R>,
    pub(super) parent_rows: Tensor<R>,
    pub(super) side: SonicSide,
    pub(super) status: Tensor<R>,
    pub(super) dry_residual: CurveValue<R>,
    pub(super) source_residual: CurveValue<R>,
    pub(super) endpoint_points: usize,
    pub(super) bracket_points: usize,
    pub(super) bracket_expansions: usize,
}

pub(super) struct SourceSonicCertificates<R: Runtime<DType = DType>> {
    pub(super) invocation_rows: Tensor<R>,
    pub(super) status: [Tensor<R>; 2],
    pub(super) domains: Vec<SourceSonicCertificate<R>>,
    pub(super) evidence: Vec<SourceSonicEvidence<R>>,
    pub(super) callback_points: usize,
    pub(super) point_limit: usize,
}
