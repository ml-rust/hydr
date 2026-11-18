use super::state::{Accepted, BoundForcing, NativeEvaluation, NativeInterval};
use super::{ChannelPoolError, context_types::Context};
use crate::coupling::links::LinkError;
use crate::coupling::links::pool_ends::{self, EndHeads, EndVolumes, ReachEnds};
use crate::numerics::water_transfers::AggregationKind;
use hydr_channel::internal::ChannelCouplingTrial;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, IndexingOps, MidpointOps,
    ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + numr::ops::TensorOps<R>
        + numr::ops::NextRepresentableOps<R>
        + solvr::integrate::CertifiedSimpsonAlgorithms<R>
        + solvr::optimize::roots::CertifiedBisectionAlgorithms<R>
        + solvr::optimize::CertifiedMappedBisectionAlgorithms<R>
        + solvr::optimize::roots::CertifiedUpperBracketAlgorithms<R>
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    pub(super) fn evaluate(
        &self,
        accepted: &Accepted<R>,
        end: f64,
        forcing: &BoundForcing<R>,
        x: &Tensor<R>,
    ) -> Result<NativeEvaluation<R>, ChannelPoolError> {
        self.check_packet(x)?;
        self.check_epochs(&accepted.reaches, &accepted.pools)?;
        let c = &self.client;
        let p = self.setup.pools.len();
        let EndHeads {
            midpoint,
            heads: connection_heads,
        } = pool_ends::end_heads(
            c,
            &accepted.pools.stages_tensor()?,
            &x.narrow(0, 0, p)?,
            &self.topology.connection_pool_indices,
        )
        .map_err(LinkError::channel_pool)?;
        let mut reaches = Vec::with_capacity(accepted.reaches.len());
        let mut endpoint_volumes = Vec::with_capacity(accepted.reaches.len());
        let mut endpoint_momentum = Vec::with_capacity(accepted.reaches.len());
        for (index, reach) in accepted.reaches.iter().enumerate() {
            let ends = ReachEnds::new(
                &connection_heads,
                &self.topology.entrance_loss_coefficients,
                self.topology.endpoint_connections[index],
            )
            .map_err(LinkError::channel_pool)?;
            let result = reach
                .trial_coupled(
                    end,
                    &forcing.reaches[index],
                    ends.operands(0),
                    ends.operands(1),
                )
                .map_err(|source| ChannelPoolError::Channel {
                    reach_id: self.setup.reaches[index].id.clone(),
                    source,
                })?;
            let next = match result {
                ChannelCouplingTrial::Accepted(next) => *next,
                ChannelCouplingTrial::Shorter {
                    suggested_end_s,
                    reason,
                } => {
                    return Err(ChannelPoolError::RecomputeInterval {
                        start_s: accepted.pools.time(),
                        requested_end_s: end,
                        suggested_end_s,
                        reason,
                    });
                }
            };
            let endpoints =
                next.coupling_endpoints()
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: self.setup.reaches[index].id.clone(),
                        source,
                    })?;
            endpoint_volumes.push(endpoints.outward_volume_m3);
            endpoint_momentum.push(endpoints.momentum_impulse_m4_per_s);
            reaches.push(next);
        }
        let EndVolumes {
            signed,
            external,
            momentum,
        } = pool_ends::end_volumes(
            c,
            &endpoint_volumes,
            &endpoint_momentum,
            &self.topology.connection_endpoint_indices,
            &self.topology.external_endpoint_indices,
        )
        .map_err(LinkError::channel_pool)?;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        let incoming = self
            .topology
            .pool_transfers
            .aggregate(c, &c.maximum(&signed, &zero)?, AggregationKind::Water)
            .map_err(ChannelPoolError::from_transfer)?;
        let outgoing = self
            .topology
            .pool_transfers
            .aggregate(
                c,
                &c.maximum(&c.neg(&signed)?, &zero)?,
                AggregationKind::Water,
            )
            .map_err(ChannelPoolError::from_transfer)?;
        let available = accepted
            .pools
            .coupling_cut_availability(&self.cut_plan)?
            .available_lower_m3;
        let demand = self.cut_demand(&signed)?;
        self.check_cuts(&available, &demand)?;
        let pools = accepted
            .pools
            .trial_to_tensors(end, &incoming.value, &outgoing.value)?
            .into_next();
        self.check_support(&accepted.pools, &pools, &signed)?;
        self.check_epochs(&reaches, &pools)?;
        let output = self.packet(&pools, &signed)?;
        let residuals = self.residuals(x, &output)?;
        Ok(NativeEvaluation {
            reaches,
            pools,
            interval: NativeInterval {
                input_packet: x.clone(),
                output_packet: output,
                midpoint_pool_stages_m: midpoint,
                connection_volume_m3: signed,
                connection_momentum_m4_per_s: momentum,
                external_volume_m3: external,
                cut_available_lower_m3: available,
                cut_outgoing_upper_m3: demand,
                pool_incoming_bound_m3: incoming.bound,
                pool_outgoing_bound_m3: outgoing.bound,
                head_residual_m: residuals.head,
                transfer_residual_m3: residuals.transfer,
            },
        })
    }
}
