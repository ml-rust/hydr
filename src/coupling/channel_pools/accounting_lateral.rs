use super::{
    ChannelPoolError,
    accounting_math::{add_bounds, nonnegative, zero},
    context_types::Context,
    reductions::{reduce, upper_sum},
};
use crate::channel::ChannelReach;
use crate::numerics::water_transfers::reject;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, TwoSumOps,
    UnaryOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

type LateralTotals<R> = (Tensor<R>, Tensor<R>, Tensor<R>);

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + TwoSumOps<R>
        + UnaryOps<R>,
{
    pub(super) fn lateral_totals(
        &self,
        reaches: &[ChannelReach<R>],
        interval: bool,
    ) -> Result<LateralTotals<R>, ChannelPoolError> {
        let c = &self.client;
        let mut values = Vec::with_capacity(reaches.len());
        let mut bounds = Vec::with_capacity(reaches.len());
        for reach in reaches {
            let view = reach.coupling_accounting();
            nonnegative(c, view.lateral_inflow_mask, "native lateral direction mask")?;
            reject(
                c,
                &c.mul(
                    &c.ne(view.lateral_inflow_mask, &zero(c)?)?,
                    &c.ne(view.lateral_inflow_mask, &c.add_scalar(&zero(c)?, 1.0)?)?,
                )?,
                "lateral direction mask requires binary values",
            )?;
            let original = if interval {
                view.lateral_interval_volume_m3.ok_or_else(|| ChannelPoolError::numerical("channel interval lateral receipt is missing. Retain the original accepted transfer operands"))?
            } else {
                view.lateral_volume_m3
            };
            values.push(original.clone());
            if !interval {
                bounds.push(view.lateral_counter_roundoff_bound_m3.clone());
            }
        }
        let packed = pack(c, &values)?;
        let (totals, report_bounds) = reduce(c, &self.reductions.lateral, &packed)?;
        let mut total_bound = add_bounds(
            c,
            &[
                &report_bounds.narrow(0, 0, 1)?,
                &report_bounds.narrow(0, 1, 1)?,
            ],
        )?;
        if !interval {
            let additions = upper_sum(c, &self.reductions.lateral, &pack(c, &bounds)?)?;
            total_bound = add_bounds(
                c,
                &[
                    &total_bound,
                    &additions.narrow(0, 0, 1)?,
                    &additions.narrow(0, 1, 1)?,
                ],
            )?;
        }
        Ok((
            totals.narrow(0, 0, 1)?,
            totals.narrow(0, 1, 1)?,
            total_bound,
        ))
    }
}

pub(super) fn pack<R: Runtime<DType = DType>>(
    client: &R::Client,
    values: &[Tensor<R>],
) -> Result<Tensor<R>, ChannelPoolError>
where
    R::Client: ShapeOps<R>,
{
    if values.iter().all(|value| value.numel() == 0) {
        return Ok(Tensor::zeros(&[0], DType::F64, client.device())?);
    }
    Ok(client.cat(&values.iter().collect::<Vec<_>>(), 0)?)
}
