//! Private finite-reservoir access with owned forcing and exact common intervals.

use super::{
    ChannelBoundary, ChannelError, ChannelLateralInput, ChannelReach, ChannelTensorBoundary,
    ChannelTensorLateralInput,
    conserved_update::{exact, expanded, magnitude, transfer},
    forcing::{BoundChannelForcing, BoundLateralInputs},
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

/// External policies and lateral operands copied once from caller storage.
#[derive(Debug)]
#[doc(hidden)]
pub struct ChannelCouplingForcing<R: Runtime<DType = DType>> {
    start_s: f64,
    end_s: f64,
    left: Option<ChannelTensorBoundary<R>>,
    right: Option<ChannelTensorBoundary<R>>,
    lateral: BoundLateralInputs<R>,
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

impl<R: Runtime<DType = DType>> ChannelReach<R> {
    #[doc(hidden)]
    pub fn coupling_accounting(&self) -> ChannelCouplingAccounting<'_, R> {
        ChannelCouplingAccounting {
            cell_volume_m3: &self.accepted.volume.high,
            cell_volume_low_m3: &self.accepted.volume.low,
            lateral_volume_m3: &self.accepted.lateral.cumulative_volume,
            lateral_interval_volume_m3: self
                .accepted
                .last
                .as_ref()
                .map(|last| &last.lateral_receipt.volume),
            lateral_inflow_mask: &self.context.lateral.inflow_mask,
            lateral_counter_roundoff_bound_m3: &self.accepted.lateral.counter_roundoff_bound,
            face_transfer_underflow_bound_m3: self
                .accepted
                .last
                .as_ref()
                .map(|last| &last.transfer_underflow_bound),
        }
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::NextRepresentableOps<R>,
{
    /// Bound on the water the stored interval update differs from its exact expansion update.
    #[doc(hidden)]
    pub fn coupling_water_assembly_bound_m3(&self) -> Result<Option<Tensor<R>>, ChannelError> {
        self.accepted
            .last
            .as_ref()
            .map(|last| {
                let c = &self.context.client;
                let count = self.context.count();
                let left = last.face_volume.narrow(0, 0, count)?;
                let right = last.face_volume.narrow(0, 1, count)?;
                // The stored update used these rounded values.
                let delta = c.sub(&left, &right)?;
                self.context
                    .finite(&delta, "channel transport water delta")?;
                let stored = (!last.forcing.lateral.has_nonzero_values)
                    .then(|| c.add(&last.previous_volume.high, &delta))
                    .transpose()?;
                if let Some(stored) = &stored {
                    self.context
                        .finite(stored, "channel transported water volume")?;
                }
                let (delta_error, stored_error) =
                    expanded(c, "channel transport water assembly", |a| {
                        let faces = exact(a, &last.face_volume, &last.face_volume_low)?;
                        let transfer = transfer(a, &faces, count)?;
                        let rounded = a.expansion_point(&delta)?;
                        let difference = a.expansion_sub(&transfer, &rounded)?;
                        let delta_error = magnitude(a, &difference)?;
                        let stored_error = match &stored {
                            Some(stored) => {
                                let previous = exact(
                                    a,
                                    &last.previous_volume.high,
                                    &last.previous_volume.low,
                                )?;
                                let update = a.expansion_add(&previous, &transfer)?;
                                let stored = a.expansion_point(stored)?;
                                let difference = a.expansion_sub(&update, &stored)?;
                                Some(magnitude(a, &difference)?)
                            }
                            None => None,
                        };
                        Ok((delta_error, stored_error))
                    })?;
                self.context
                    .finite(&delta_error, "channel transport subtraction bound")?;
                match stored_error {
                    Some(bound) => {
                        self.context
                            .finite(&bound, "channel water addition bound")?;
                        Ok(bound)
                    }
                    None => {
                        self.context.finite(
                            &last.lateral_water_assembly_bound,
                            "channel lateral water assembly bound",
                        )?;
                        self.context
                            .bound_add(&delta_error, &last.lateral_water_assembly_bound)
                    }
                }
            })
            .transpose()
    }
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

    fn bind_coupling_owned(
        &self,
        start_s: f64,
        end_s: f64,
        left: Option<ChannelTensorBoundary<R>>,
        right: Option<ChannelTensorBoundary<R>>,
        lateral: Vec<ChannelTensorLateralInput<R>>,
    ) -> Result<ChannelCouplingForcing<R>, ChannelError> {
        if matches!(
            &self.definition().geometry,
            super::ChannelGeometry::PrismaticSurveyed { .. }
        ) {
            return Err(ChannelError::input(
                "surveyed coordinator attachment is unsupported: advance the reach with Wall or complete Exterior boundaries",
            ));
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
        })
    }
}

impl<R: Runtime<DType = DType>> ChannelReach<R>
where
    R::Client: numr::ops::TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + numr::ops::NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + BatchedBisectionAlgorithms<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn trial_coupled(
        &self,
        requested_end_s: f64,
        forcing: &ChannelCouplingForcing<R>,
        left: Option<ChannelReservoirOperands<'_, R>>,
        right: Option<ChannelReservoirOperands<'_, R>>,
    ) -> Result<ChannelCouplingTrial<R>, ChannelError> {
        let endpoint = |external: &Option<ChannelTensorBoundary<R>>,
                        reservoir: Option<ChannelReservoirOperands<'_, R>>|
         -> Result<ChannelTensorBoundary<R>, ChannelError> {
            match (external, reservoir) {
                (Some(external), None) => Ok(external.clone()),
                (None, Some(reservoir)) => Ok(ChannelTensorBoundary::QuiescentReservoir {
                    elevation_m: reservoir.elevation_m.clone(),
                    entrance_loss_coefficient: reservoir.entrance_loss_coefficient.clone(),
                }
                .into_owned(&self.context.client)?),
                _ => Err(ChannelError::input(
                    "endpoint ownership: supply exactly one external policy or connected reservoir operand",
                )),
            }
        };
        let left = endpoint(&forcing.left, left)?;
        let right = endpoint(&forcing.right, right)?;
        self.context.lateral.check_endpoints(
            &self.context.client,
            forcing.start_s,
            forcing.end_s,
            Some(&left),
            Some(&right),
        )?;
        let bound = BoundChannelForcing {
            definition: Arc::clone(&forcing.lateral.definition),
            start_s: forcing.start_s,
            end_s: forcing.end_s,
            left,
            right,
            lateral: forcing.lateral.clone(),
        };
        match self.step_bound_policy(requested_end_s, &bound, true) {
            Ok(step) => Ok(ChannelCouplingTrial::Accepted(Box::new(step.next))),
            Err(ChannelError::RecomputeInterval {
                suggested_end_s,
                reason,
                ..
            }) if !forcing.has_interval_volumes() => Ok(ChannelCouplingTrial::Shorter {
                suggested_end_s,
                reason,
            }),
            Err(error) => Err(error),
        }
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
#[path = "coupling_tests.rs"]
mod tests;
