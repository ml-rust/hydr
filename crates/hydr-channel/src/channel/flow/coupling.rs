//! Private finite-reservoir access with owned forcing and exact common intervals.

mod accounting;
#[cfg(test)]
mod fixture;
mod trial;

use super::{
    ChannelBoundary, ChannelError, ChannelLateralInput, ChannelReach, ChannelTensorBoundary,
    ChannelTensorLateralInput, forcing::BoundLateralInputs, interval_start::ReservedStart,
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, TwoSumOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::optimize::roots::BatchedBisectionAlgorithms;
use std::sync::Arc;

/// Lateral boundary identity prefix reserved for coordinator surface exchange.
///
/// Caller definitions and caller forcing reject it. `ChannelReach::new_coordinated` admits it.
#[doc(hidden)]
pub const SURFACE_EXCHANGE_BINDING_PREFIX: &str = "surface-exchange:";

/// Reject a caller-supplied lateral boundary identity that uses the reserved prefix.
pub(super) fn reject_reserved_boundary(boundary_id: &str) -> Result<(), ChannelError> {
    if boundary_id.starts_with(SURFACE_EXCHANGE_BINDING_PREFIX) {
        return Err(ChannelError::input(format!(
            "boundary_id={boundary_id:?} uses the reserved prefix {SURFACE_EXCHANGE_BINDING_PREFIX:?}: rename the lateral binding; surface exchange belongs to the surface-channel coordinator"
        )));
    }
    Ok(())
}

/// Endpoint name for coordinator endpoint errors. `None` is a coordinator-owned connection.
fn endpoint_name<R: Runtime<DType = DType>>(endpoint: Option<&ChannelTensorBoundary<R>>) -> &str {
    match endpoint {
        None => "connected",
        Some(ChannelTensorBoundary::Wall) => "Wall",
        Some(ChannelTensorBoundary::Exterior { .. }) => "Exterior",
        Some(ChannelTensorBoundary::Stage { .. }) => "Stage",
        Some(ChannelTensorBoundary::QuiescentReservoir { .. }) => "QuiescentReservoir",
        Some(ChannelTensorBoundary::Discharge { .. }) => "Discharge",
        Some(ChannelTensorBoundary::FluxPort(_)) => "FluxPort",
    }
}

/// Admit a surveyed reach into a coordinator only through lateral bindings.
///
/// Both endpoints must hold an external `Wall` or `Exterior` policy.
/// A connected endpoint or a scalar policy rejects.
fn check_lateral_only<R: Runtime<DType = DType>>(
    left: Option<&ChannelTensorBoundary<R>>,
    right: Option<&ChannelTensorBoundary<R>>,
) -> Result<(), ChannelError> {
    let admitted = |endpoint: Option<&ChannelTensorBoundary<R>>| {
        matches!(
            endpoint,
            Some(ChannelTensorBoundary::Wall | ChannelTensorBoundary::Exterior { .. })
        )
    };
    if admitted(left) && admitted(right) {
        return Ok(());
    }
    Err(ChannelError::input(format!(
        "surveyed coordinator endpoints left={}, right={}: surveyed reaches join a coordinator through lateral bindings only; supply Wall or complete Exterior at both endpoints",
        endpoint_name(left),
        endpoint_name(right)
    )))
}

/// External policies and lateral operands copied once from caller storage.
#[derive(Debug)]
#[doc(hidden)]
pub struct ChannelCouplingForcing<R: Runtime<DType = DType>> {
    pub(super) start_s: f64,
    pub(super) end_s: f64,
    /// `None` marks a coordinator-owned reservoir endpoint.
    pub(super) left: Option<ChannelTensorBoundary<R>>,
    pub(super) right: Option<ChannelTensorBoundary<R>>,
    pub(super) lateral: BoundLateralInputs<R>,
    /// Accepted-start operands from a reservation. `None` calculates them in the trial.
    pub(super) reserved: Option<Arc<ReservedStart<R>>>,
}

/// Actual coordinator-owned midpoint reservoir operands. Neither operand is projected.
#[doc(hidden)]
pub struct ChannelReservoirOperands<'a, R: Runtime<DType = DType>> {
    pub elevation_m: &'a Tensor<R>,
    pub entrance_loss_coefficient: &'a Tensor<R>,
}

#[doc(hidden)]
pub enum ChannelCouplingTrial<R: Runtime<DType = DType>> {
    Accepted(Box<ChannelReach<R>>),
    Shorter {
        suggested_end_s: f64,
        reason: String,
    },
}

/// Checkpoint projection retains absent coordinator-owned endpoints.
#[doc(hidden)]
pub struct ChannelCouplingExternalData {
    pub start_s: f64,
    pub end_s: f64,
    pub left: Option<ChannelBoundary>,
    pub right: Option<ChannelBoundary>,
    pub lateral_inputs: Vec<ChannelLateralInput>,
}

/// Owned native endpoint operands in `[left, right]` order.
#[doc(hidden)]
pub struct ChannelCouplingEndpoints<R: Runtime<DType = DType>> {
    /// Positive channel-to-reservoir volume, from stored shared face volumes.
    pub outward_volume_m3: Tensor<R>,
    /// Actual contribution to longitudinal channel momentum: left positive, right negative.
    /// Includes the interior-side pressure impulse with the existing face convention.
    pub momentum_impulse_m4_per_s: Tensor<R>,
}

/// Borrowed native accounting operands for the parent's cached reductions.
#[doc(hidden)]
pub struct ChannelCouplingAccounting<'a, R: Runtime<DType = DType>> {
    pub cell_volume_m3: &'a Tensor<R>,
    /// Low parts of the cell volumes. The storage inventory requires them to be zero.
    pub cell_volume_low_m3: &'a Tensor<R>,
    pub lateral_volume_m3: &'a Tensor<R>,
    pub lateral_interval_volume_m3: Option<&'a Tensor<R>>,
    pub lateral_inflow_mask: &'a Tensor<R>,
    pub lateral_counter_roundoff_bound_m3: &'a Tensor<R>,
    pub face_transfer_underflow_bound_m3: Option<&'a Tensor<R>>,
}

impl<R: Runtime<DType = DType>> ChannelCouplingForcing<R> {
    #[doc(hidden)]
    pub fn has_interval_volumes(&self) -> bool {
        self.lateral.has_interval_volumes
    }

    /// Project only when recording explicit checkpoint data.
    #[doc(hidden)]
    pub fn project_external(
        &self,
        client: &R::Client,
    ) -> Result<ChannelCouplingExternalData, ChannelError> {
        Ok(ChannelCouplingExternalData {
            start_s: self.start_s,
            end_s: self.end_s,
            left: self
                .left
                .as_ref()
                .map(ChannelTensorBoundary::project)
                .transpose()?,
            right: self
                .right
                .as_ref()
                .map(ChannelTensorBoundary::project)
                .transpose()?,
            lateral_inputs: self.lateral.project(client)?,
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn bind_coupling_host(
        &self,
        start_s: f64,
        end_s: f64,
        left: Option<&ChannelBoundary>,
        right: Option<&ChannelBoundary>,
        lateral: &[ChannelLateralInput],
    ) -> Result<ChannelCouplingForcing<R>, ChannelError> {
        let c = &self.context.client;
        self.bind_coupling_owned(
            start_s,
            end_s,
            left.map(|boundary| ChannelTensorBoundary::from_host(c, boundary))
                .transpose()?,
            right
                .map(|boundary| ChannelTensorBoundary::from_host(c, boundary))
                .transpose()?,
            self.context.lateral.host_inputs(c, lateral)?,
        )
    }

    #[doc(hidden)]
    pub fn bind_coupling_tensor(
        &self,
        start_s: f64,
        end_s: f64,
        left: Option<&ChannelTensorBoundary<R>>,
        right: Option<&ChannelTensorBoundary<R>>,
        lateral: &[ChannelTensorLateralInput<R>],
    ) -> Result<ChannelCouplingForcing<R>, ChannelError> {
        let c = &self.context.client;
        self.bind_coupling_owned(
            start_s,
            end_s,
            left.map(|boundary| boundary.clone().into_owned(c))
                .transpose()?,
            right
                .map(|boundary| boundary.clone().into_owned(c))
                .transpose()?,
            lateral
                .iter()
                .map(|input| input.clone().into_owned(c))
                .collect::<Result<_, _>>()?,
        )
    }

    pub(super) fn bind_coupling_owned(
        &self,
        start_s: f64,
        end_s: f64,
        left: Option<ChannelTensorBoundary<R>>,
        right: Option<ChannelTensorBoundary<R>>,
        lateral: Vec<ChannelTensorLateralInput<R>>,
    ) -> Result<ChannelCouplingForcing<R>, ChannelError> {
        if self.context.surveyed() {
            check_lateral_only(left.as_ref(), right.as_ref())?;
        }
        super::definition::check_forcing_interval(self.time(), end_s, start_s, end_s)?;
        self.context.lateral.check_endpoints(
            &self.context.client,
            start_s,
            end_s,
            left.as_ref(),
            right.as_ref(),
        )?;
        Ok(ChannelCouplingForcing {
            start_s,
            end_s,
            left,
            right,
            lateral: self
                .context
                .lateral
                .bind_lateral_owned(&self.context.client, lateral)?,
            reserved: None,
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R> + ShapeOps<R> + ScalarOps<R> + UnaryOps<R>,
{
    #[doc(hidden)]
    pub fn coupling_endpoints(&self) -> Result<ChannelCouplingEndpoints<R>, ChannelError> {
        let last = self.accepted.last.as_ref().ok_or_else(|| {
            ChannelError::input("endpoint transfers require an accepted interval: calculate a complete coupled trial")
        })?;
        let c = &self.context.client;
        let count = self.context.count();
        let left = last.face_volume.narrow(0, 0, 1)?;
        let right = last.face_volume.narrow(0, count, 1)?;
        let outward_volume_m3 = c.cat(&[&c.neg(&left)?, &right], 0)?;
        let left_momentum = c.add(
            &last.face_momentum.narrow(0, 0, 1)?,
            &last.right_pressure.narrow(0, 0, 1)?,
        )?;
        let right_momentum = c.neg(&c.add(
            &last.face_momentum.narrow(0, count, 1)?,
            &last.left_pressure.narrow(0, count, 1)?,
        )?)?;
        Ok(ChannelCouplingEndpoints {
            outward_volume_m3,
            momentum_impulse_m4_per_s: c.cat(&[&left_momentum, &right_momentum], 0)?,
        })
    }
}

#[cfg(test)]
#[path = "coupling_surveyed_tests.rs"]
mod surveyed_tests;
