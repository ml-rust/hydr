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
mod tests {
    use super::super::{
        ChannelBoundary, ChannelEnd, ChannelFluxPort, ChannelGeometry, ChannelLateralBinding,
        ChannelLateralDirection, ChannelLateralInput, ChannelReach,
        surveyed_update::tests::{TestResult, cpu, exact_sum, moving_reach},
    };
    use super::{ChannelCouplingTrial, ChannelReservoirOperands, SURFACE_EXCHANGE_BINDING_PREFIX};
    use crate::channel::CrossSectionPoint;
    use crate::forcing::WaterAmount;
    use numr::runtime::RuntimeClient;
    use numr::runtime::cpu::CpuRuntime;
    use numr::tensor::Tensor;

    /// Coordinated surveyed reach with one reserved withdrawal binding at the middle cell.
    fn coordinated() -> TestResult<ChannelReach<CpuRuntime>> {
        let standalone =
            moving_reach::<CpuRuntime>(cpu(), &[10.0, 6.0, 0.0], &[0.0; 3], &[0.0; 3])?;
        let mut definition = standalone.definition().clone();
        definition.lateral_bindings = vec![ChannelLateralBinding {
            boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}bank:withdrawal"),
            cell_id: "cell-1".into(),
            direction: ChannelLateralDirection::Withdrawal,
            ownership_assertion: "coordinator surface exchange".into(),
        }];
        Ok(ChannelReach::new_coordinated(
            cpu(),
            definition,
            standalone.context.original_initial.clone(),
        )?)
    }

    fn withdrawal(rate_m3_per_s: f64) -> ChannelLateralInput {
        ChannelLateralInput::Withdrawal {
            boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}bank:withdrawal"),
            amount: WaterAmount::Rate { rate_m3_per_s },
        }
    }

    #[test]
    fn surveyed_lateral_only_coordination_advances() -> TestResult {
        let reach = coordinated()?;
        let wall = ChannelBoundary::Wall;
        let exterior = ChannelBoundary::Exterior {
            depth_m: 0.0,
            velocity_m_per_s: 0.0,
        };
        let inputs = [withdrawal(0.5)];
        let forcing = reach.bind_coupling_host(0.0, 0.05, Some(&wall), Some(&exterior), &inputs)?;
        let reservation = reach.coupling_reservation(&forcing)?;
        let donor = reservation.donor_volume_m3.try_to_vec::<f64>()?;
        let stored = reach
            .coupling_accounting()
            .cell_volume_m3
            .try_to_vec::<f64>()?;
        assert!(donor.iter().zip(&stored).all(|(d, s)| d <= s && *d >= 0.0));
        assert!(reservation.start_cfl_s > 0.0);

        // Stage is the section minimum plus the certified recovered depth.
        let (stage, wet) = reach.coupling_cell_stages(&[2, 0, 1])?;
        let stage = stage.try_to_vec::<f64>()?;
        let wet = wet.try_to_vec::<f64>()?;
        let state = reach.state()?;
        for (slot, cell) in [2, 0, 1].into_iter().enumerate() {
            let result = &state[cell];
            assert_eq!(wet[slot], if result.dry { 0.0 } else { 1.0 });
            let expected = result.stage_m.unwrap_or(0.0);
            assert_eq!(stage[slot].to_bits(), expected.to_bits(), "cell={cell}");
        }

        let next = match reach.trial_coupled(0.05, &forcing, None, None)? {
            ChannelCouplingTrial::Accepted(next) => next,
            ChannelCouplingTrial::Shorter { reason, .. } => {
                return Err(format!("a short surveyed interval must accept: {reason}").into());
            }
        };
        let ledger = next.lateral_ledgers()?;
        assert_eq!(ledger.len(), 1);
        assert!(next.ledger().lateral_out_m3 > 0.0);
        Ok(())
    }

    /// Coordinated survey of the given cell volumes, with a profile of the given points.
    fn coordinated_survey(
        volumes: &[f64],
        points: &[(f64, f64)],
    ) -> TestResult<ChannelReach<CpuRuntime>> {
        let standalone = moving_reach::<CpuRuntime>(
            cpu(),
            volumes,
            &vec![0.0; volumes.len()],
            &vec![0.0; volumes.len()],
        )?;
        let mut definition = standalone.definition().clone();
        if let ChannelGeometry::PrismaticSurveyed { profile, .. } = &mut definition.geometry {
            profile.points = points
                .iter()
                .map(|&(station_m, elevation_m)| CrossSectionPoint {
                    station_m,
                    elevation_m,
                })
                .collect();
        }
        definition.lateral_bindings = vec![ChannelLateralBinding {
            boundary_id: format!("{SURFACE_EXCHANGE_BINDING_PREFIX}bank:withdrawal"),
            cell_id: "cell-1".into(),
            direction: ChannelLateralDirection::Withdrawal,
            ownership_assertion: "coordinator surface exchange".into(),
        }];
        Ok(ChannelReach::new_coordinated(
            cpu(),
            definition,
            standalone.context.original_initial.clone(),
        )?)
    }

    const RECTANGLE: [(f64, f64); 4] = [(-1.0, 3.5), (-1.0, 0.0), (1.0, 0.0), (1.0, 3.5)];
    const TRIANGLE: [(f64, f64); 3] = [(-1.0, 3.5), (0.0, 0.0), (1.0, 3.5)];

    fn closed_port() -> TestResult<ChannelBoundary> {
        Ok(ChannelBoundary::FluxPort(ChannelFluxPort::new(
            0.0,
            0.0,
            10.0,
            Some(10.0),
            "closed test port",
        )?))
    }

    #[test]
    fn surveyed_none_end_steps_and_conserves_water() -> TestResult {
        let reach = coordinated_survey(&[10.0, 8.0, 6.0], &RECTANGLE)?;
        let before = exact_sum(
            [&reach.accepted.volume.high, &reach.accepted.volume.low]
                .into_iter()
                .map(|part| part.try_to_vec::<f64>())
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .flatten(),
        );
        let client = cpu();
        let elevation = Tensor::from_slice(&[0.5], &[1], client.device())?;
        let loss = Tensor::from_slice(&[0.0], &[1], client.device())?;
        let wall = ChannelBoundary::Wall;
        let forcing = reach.bind_coupling_host(0.0, 0.05, Some(&wall), None, &[withdrawal(0.0)])?;
        let next = match reach.trial_coupled(
            0.05,
            &forcing,
            None,
            Some(ChannelReservoirOperands {
                elevation_m: &elevation,
                entrance_loss_coefficient: &loss,
            }),
        )? {
            ChannelCouplingTrial::Accepted(next) => next,
            ChannelCouplingTrial::Shorter { reason, .. } => {
                return Err(format!("a short surveyed interval must accept: {reason}").into());
            }
        };
        let outward = next
            .coupling_endpoints()?
            .outward_volume_m3
            .try_to_vec::<f64>()?;
        assert_eq!(outward[0], 0.0, "the wall end exchanges no water");
        assert!(outward[1] > 0.0, "the reservoir below the end stage drains");
        let data = next.snapshot_data()?;
        let after = exact_sum(
            data.volume_m3
                .iter()
                .chain(&data.volume_low_m3)
                .copied()
                .chain(before.iter().map(|part| -part)),
        );
        let change: f64 = after.iter().sum();
        let ledger = &data.ledger;
        let boundary = ledger.boundary_in_m3 - ledger.boundary_out_m3;
        assert!(
            (boundary + outward[1]).abs() <= f64::EPSILON * outward[1],
            "boundary={boundary:e}, outward={:e}",
            outward[1]
        );
        let bound =
            ledger.storage_selection_abs_bound_m3 + ledger.numerical_transfer_underflow_bound_m3;
        let slack = f64::EPSILON * change.abs().max(outward[1]);
        assert!(
            (change + outward[1]).abs() <= bound + slack,
            "change={change:e}, outward={:e}, bound={bound:e}",
            outward[1]
        );
        Ok(())
    }

    #[test]
    fn surveyed_rectangle_port_end_accepts() -> TestResult {
        let reach = coordinated_survey(&[10.0, 8.0, 6.0], &RECTANGLE)?;
        let state = reach.coupling_end_state(ChannelEnd::Right)?;
        assert_eq!(state.cell, 2);
        assert_eq!(state.width_m, 2.0);
        assert_eq!(state.bed_elevation_m, 0.0);
        assert_eq!(state.area_m2.try_to_vec::<f64>()?, vec![1.2]);
        assert_eq!(state.depth_m.try_to_vec::<f64>()?, vec![0.6]);
        let port = closed_port()?;
        let wall = ChannelBoundary::Wall;
        let forcing =
            reach.bind_coupling_host(0.0, 0.05, Some(&wall), Some(&port), &[withdrawal(0.0)])?;
        match reach.trial_coupled(0.05, &forcing, None, None)? {
            ChannelCouplingTrial::Accepted(_) => Ok(()),
            ChannelCouplingTrial::Shorter { reason, .. } => {
                Err(format!("a short port interval must accept: {reason}").into())
            }
        }
    }

    #[test]
    fn surveyed_non_rectangle_port_end_rejects_and_keeps_state() -> TestResult {
        let reach = coordinated_survey(&[10.0, 6.0, 2.0], &TRIANGLE)?;
        let before = reach.snapshot_data()?;
        for end in [ChannelEnd::Left, ChannelEnd::Right] {
            let error = reach
                .coupling_end_state(end)
                .err()
                .ok_or("a non-rectangular surveyed end must reject")?;
            let message = error.to_string();
            assert!(message.contains("port-section-mismatch"), "{message}");
        }
        assert_eq!(reach.snapshot_data()?, before);
        Ok(())
    }

    #[test]
    fn surveyed_free_end_admits_every_standalone_policy() -> TestResult {
        let reach = coordinated_survey(&[10.0, 8.0, 6.0], &RECTANGLE)?;
        let policies = [
            ChannelBoundary::Wall,
            ChannelBoundary::Exterior {
                depth_m: 0.5,
                velocity_m_per_s: 0.0,
            },
            ChannelBoundary::Stage { elevation_m: 0.5 },
            ChannelBoundary::Discharge {
                discharge_m3_per_s: 0.0,
            },
            ChannelBoundary::QuiescentReservoir {
                elevation_m: 0.5,
                entrance_loss_coefficient: 0.0,
            },
            closed_port()?,
        ];
        let inputs = [withdrawal(0.0)];
        for policy in &policies {
            reach
                .bind_coupling_host(0.0, 0.05, None, Some(policy), &inputs)
                .map_err(|error| format!("{policy:?} at the right end: {error}"))?;
            reach
                .bind_coupling_host(0.0, 0.05, Some(policy), None, &inputs)
                .map_err(|error| format!("{policy:?} at the left end: {error}"))?;
        }
        Ok(())
    }
}
