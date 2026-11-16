use super::state::{Accepted, BoundForcing, NativeEvaluation, NativeInterval};
use super::{ChannelPoolError, context_types::Context};
use crate::channel::{ChannelCouplingTrial, ChannelReservoirOperands};
use crate::numerics::water_transfers::{AggregationKind, finite};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
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
        let midpoint = c.midpoint(&accepted.pools.stages_tensor()?, &x.narrow(0, 0, p)?)?;
        finite(c, &midpoint, "used midpoint pool stages")?;
        let connection_heads =
            c.index_select(&midpoint, 0, &self.topology.connection_pool_indices)?;
        let mut reaches = Vec::with_capacity(accepted.reaches.len());
        let mut endpoint_volumes = Vec::with_capacity(accepted.reaches.len());
        let mut endpoint_momentum = Vec::with_capacity(accepted.reaches.len());
        for (index, reach) in accepted.reaches.iter().enumerate() {
            let attachments = self.topology.endpoint_connections[index];
            let gather = |values: &Tensor<R>| -> Result<[Option<Tensor<R>>; 2], ChannelPoolError> {
                Ok([
                    attachments[0].map(|i| values.narrow(0, i, 1)).transpose()?,
                    attachments[1].map(|i| values.narrow(0, i, 1)).transpose()?,
                ])
            };
            let heads = gather(&connection_heads)?;
            let losses = gather(&self.topology.entrance_loss_coefficients)?;
            let operands = |side: usize| {
                heads[side].as_ref().zip(losses[side].as_ref()).map(
                    |(elevation_m, entrance_loss_coefficient)| ChannelReservoirOperands {
                        elevation_m,
                        entrance_loss_coefficient,
                    },
                )
            };
            let result = reach
                .trial_coupled(end, &forcing.reaches[index], operands(0), operands(1))
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
        let volumes = c.cat(&endpoint_volumes.iter().collect::<Vec<_>>(), 0)?;
        let momenta = c.cat(&endpoint_momentum.iter().collect::<Vec<_>>(), 0)?;
        let signed = c.index_select(&volumes, 0, &self.topology.connection_endpoint_indices)?;
        let external = c.index_select(&volumes, 0, &self.topology.external_endpoint_indices)?;
        let momentum = c.index_select(&momenta, 0, &self.topology.connection_endpoint_indices)?;
        let zero = Tensor::<R>::zeros(&[1], DType::F64, c.device())?;
        let incoming = self.topology.pool_transfers.aggregate(
            c,
            &c.maximum(&signed, &zero)?,
            AggregationKind::Water,
        )?;
        let outgoing = self.topology.pool_transfers.aggregate(
            c,
            &c.maximum(&c.neg(&signed)?, &zero)?,
            AggregationKind::Water,
        )?;
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
