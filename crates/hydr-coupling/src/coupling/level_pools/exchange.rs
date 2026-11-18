use super::{
    CoupledLevelPoolError,
    context::Context,
    topology::{finite, reject, upward_add},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CorrectlyRoundedOps, CumulativeOps, IndexingOps,
    MidpointOps, ReduceOps, ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::Interp1dAlgorithms;

#[derive(Clone, Debug)]
pub(super) struct Exchange<R: Runtime<DType = DType>> {
    pub proposed: Tensor<R>,
    pub signed: Tensor<R>,
    pub factor: Tensor<R>,
    pub incoming: Tensor<R>,
    pub outgoing: Tensor<R>,
    pub aggregation_bound: f64,
}
impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CorrectlyRoundedOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    pub fn exchange(
        &self,
        proposed: Tensor<R>,
        available: &Tensor<R>,
    ) -> Result<Exchange<R>, CoupledLevelPoolError> {
        let c = &self.client;
        finite(c, &proposed, "held connection volume")?;
        let (positive, negative) = self.split(&proposed)?;
        let demands = self
            .tree
            .aggregate(c, &c.cat(&[&positive, &negative], 0)?)?;
        let upper = upward_add(c, &demands.volume, &demands.bound)?;
        let limited = c.gt(&upper, available)?;
        let margin = c.add_scalar(
            &c.mul(available, &self.tree.degree_margin)?,
            f64::from_bits(1),
        )?;
        let lower = c.sub(available, &margin)?;
        let has_water = c.gt(available, &self.zero)?;
        reject(
            c,
            &c.mul(&c.mul(&limited, &has_water)?, &c.le(&lower, &self.zero)?)?,
            "positive donor storage cannot represent a conservative shared allocation. Rescale storage",
        )?;
        let numerator = c.where_cond(&c.mul(&limited, &has_water)?, &lower, &self.zero)?;
        let denominator = c.where_cond(&limited, &upper, &self.one)?;
        let ratio = c.div(&numerator, &denominator)?;
        let rounding = c.add_scalar(
            &c.mul_scalar(&c.abs(&ratio)?, f64::EPSILON)?,
            f64::from_bits(1),
        )?;
        let conservative = c.sub(&ratio, &rounding)?;
        reject(
            c,
            &c.mul(
                &c.mul(&limited, &has_water)?,
                &c.le(&conservative, &self.zero)?,
            )?,
            "positive donor allocation factor underflows. Rescale the competing demands",
        )?;
        let limited_factor = c.where_cond(&has_water, &conservative, &self.zero)?;
        let pool_factor = c.where_cond(&limited, &limited_factor, &self.one)?;
        let fa = c.index_select(&pool_factor, 0, &self.a)?;
        let fb = c.index_select(&pool_factor, 0, &self.b)?;
        let factor = c.where_cond(&c.lt(&proposed, &self.zero)?, &fb, &fa)?;
        let signed = c.mul(&proposed, &factor)?;
        finite(c, &signed, "accepted connection volume")?;
        reject(
            c,
            &c.mul(
                &c.mul(
                    &c.gt(&c.abs(&proposed)?, &self.zero)?,
                    &c.gt(&factor, &self.zero)?,
                )?,
                &c.eq(&signed, &self.zero)?,
            )?,
            "positive accepted connection volume underflows. Rescale the exchange",
        )?;
        let (positive, negative) = self.split(&signed)?;
        let outgoing = self
            .tree
            .aggregate(c, &c.cat(&[&positive, &negative], 0)?)?;
        let incoming = self
            .tree
            .aggregate(c, &c.cat(&[&negative, &positive], 0)?)?;
        let protected = upward_add(c, &outgoing.volume, &outgoing.bound)?;
        reject(
            c,
            &c.gt(&protected, available)?,
            "represented donor totals exceed original accepted storage",
        )?;
        let aggregation_bound =
            assembly_bound::<R>(c, &incoming.bound, &outgoing.bound, self.pool_count())?;
        Ok(Exchange {
            proposed,
            signed,
            factor,
            incoming: incoming.volume,
            outgoing: outgoing.volume,
            aggregation_bound,
        })
    }
    fn split(&self, signed: &Tensor<R>) -> Result<(Tensor<R>, Tensor<R>), CoupledLevelPoolError> {
        let c = &self.client;
        Ok((
            c.where_cond(&c.gt(signed, &self.zero)?, signed, &self.zero)?,
            c.where_cond(&c.lt(signed, &self.zero)?, &c.neg(signed)?, &self.zero)?,
        ))
    }
}

fn assembly_bound<R>(
    client: &R::Client,
    incoming: &Tensor<R>,
    outgoing: &Tensor<R>,
    pool_count: usize,
) -> Result<f64, CoupledLevelPoolError>
where
    R: Runtime<DType = DType>,
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
    let combined = upward_add(client, incoming, outgoing)?;
    let bound = client.max(&combined, &[0], false)?.item::<f64>()?;
    if bound == 0.0 {
        Ok(0.0)
    } else {
        crate::numerics::outward::upward_mul(bound, pool_count as f64)
            .map_err(|e| CoupledLevelPoolError::numerical(e.to_string()))
    }
}

#[cfg(test)]
mod tests;
