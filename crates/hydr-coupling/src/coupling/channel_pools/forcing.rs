use super::batch::ChannelPoolBatch;
use super::state::BoundForcing;
use super::{ChannelPoolError, ChannelPoolForcingInterval, ChannelPoolTensorForcingInterval};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, ReduceOps, ScalarOps,
    ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{AndersonAlgorithms, BatchedBisectionAlgorithms};
use std::collections::BTreeMap;
use std::sync::Arc;

impl<R: Runtime<DType = DType>> ChannelPoolBatch<R>
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
        + Interp1dAlgorithms<R>
        + BatchedBisectionAlgorithms<R>
        + AndersonAlgorithms<R>,
{
    pub(super) fn bind_host_forcing(
        &self,
        requested: f64,
        forcing: &ChannelPoolForcingInterval,
    ) -> Result<Arc<BoundForcing<R>>, ChannelPoolError> {
        check_span(self.time(), requested, forcing.start_s, forcing.end_s)?;
        let order = self.forcing_order(forcing.reaches.iter().map(|r| r.reach_id.as_str()))?;
        let mut reaches = Vec::with_capacity(order.len());
        for (index, record) in order.into_iter().enumerate() {
            let input = &forcing.reaches[record];
            self.check_endpoint_ownership(index, input.left.is_some(), input.right.is_some())?;
            let bound = self.accepted.reaches[index]
                .bind_coupling_host(
                    forcing.start_s,
                    forcing.end_s,
                    input.left.as_ref(),
                    input.right.as_ref(),
                    &input.lateral_inputs,
                )
                .map_err(|source| ChannelPoolError::Channel {
                    reach_id: input.reach_id.clone(),
                    source,
                })?;
            reaches.push(bound);
        }
        let requires_exact_span = reaches.iter().any(|r| r.has_interval_volumes());
        if requires_exact_span && forcing.end_s.to_bits() != requested.to_bits() {
            return Err(ChannelPoolError::input(
                "integrated forcing endpoint differs from requested endpoint: supply the exact original span",
            ));
        }
        Ok(Arc::new(BoundForcing {
            start_s: forcing.start_s,
            end_s: forcing.end_s,
            reaches,
            requires_exact_span,
        }))
    }
    pub(super) fn bind_tensor_forcing(
        &self,
        requested: f64,
        forcing: &ChannelPoolTensorForcingInterval<R>,
    ) -> Result<Arc<BoundForcing<R>>, ChannelPoolError> {
        check_span(self.time(), requested, forcing.start_s, forcing.end_s)?;
        let order = self.forcing_order(forcing.reaches.iter().map(|r| r.reach_id.as_str()))?;
        let mut reaches = Vec::with_capacity(order.len());
        for (index, record) in order.into_iter().enumerate() {
            let input = &forcing.reaches[record];
            self.check_endpoint_ownership(index, input.left.is_some(), input.right.is_some())?;
            reaches.push(
                self.accepted.reaches[index]
                    .bind_coupling_tensor(
                        forcing.start_s,
                        forcing.end_s,
                        input.left.as_ref(),
                        input.right.as_ref(),
                        &input.lateral_inputs,
                    )
                    .map_err(|source| ChannelPoolError::Channel {
                        reach_id: input.reach_id.clone(),
                        source,
                    })?,
            );
        }
        let requires_exact_span = reaches.iter().any(|r| r.has_interval_volumes());
        if requires_exact_span && forcing.end_s.to_bits() != requested.to_bits() {
            return Err(ChannelPoolError::input(
                "integrated forcing endpoint differs from requested endpoint: supply the exact original span",
            ));
        }
        Ok(Arc::new(BoundForcing {
            start_s: forcing.start_s,
            end_s: forcing.end_s,
            reaches,
            requires_exact_span,
        }))
    }
    fn forcing_order<'a>(
        &self,
        ids: impl Iterator<Item = &'a str>,
    ) -> Result<Vec<usize>, ChannelPoolError> {
        let mut records = BTreeMap::new();
        for (index, id) in ids.enumerate() {
            if !self.context.topology.reach_by_id.contains_key(id)
                || records.insert(id, index).is_some()
            {
                return Err(ChannelPoolError::input(format!(
                    "forcing reach_id={id:?}: supply each declared reach exactly once"
                )));
            }
        }
        self.setup()
            .reaches
            .iter()
            .map(|r| {
                records.get(r.id.as_str()).copied().ok_or_else(|| {
                    ChannelPoolError::input(format!(
                        "forcing reach_id={:?} is absent: supply complete reach records",
                        r.id
                    ))
                })
            })
            .collect()
    }
    fn check_endpoint_ownership(
        &self,
        reach: usize,
        left: bool,
        right: bool,
    ) -> Result<(), ChannelPoolError> {
        for (side, present) in [left, right].into_iter().enumerate() {
            if present == self.context.topology.endpoint_connections[reach][side].is_some() {
                return Err(ChannelPoolError::input(format!(
                    "reach_id={:?}, endpoint={side}: supply None for connected endpoints and Some for external endpoints",
                    self.setup().reaches[reach].id
                )));
            }
        }
        Ok(())
    }
}
pub(super) fn check_span(
    start: f64,
    requested: f64,
    forcing_start: f64,
    forcing_end: f64,
) -> Result<(), ChannelPoolError> {
    if !requested.is_finite()
        || requested <= start
        || !forcing_start.is_finite()
        || forcing_start.to_bits() != start.to_bits()
        || !forcing_end.is_finite()
        || forcing_end < requested
    {
        return Err(ChannelPoolError::input(format!(
            "accepted={start}, requested={requested}, forcing=[{forcing_start},{forcing_end}]: supply exact accepted start and complete finite endpoint coverage"
        )));
    }
    Ok(())
}
