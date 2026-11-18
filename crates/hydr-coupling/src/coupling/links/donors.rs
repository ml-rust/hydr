//! Donor-limited link volumes: one shared factor per donor group, outward rounded.

use super::{LinkError, LinkOps};
use crate::numerics::outward::upward_mul;
use crate::numerics::water_transfers::{self, AggregationKind, TransferTree};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Rejection texts of one coordinator's donor limit.
pub(crate) struct DonorTexts {
    /// Positive donor water cannot hold a conservative shared allocation.
    pub unrepresentable: &'static str,
    /// A constrained donor's factor rounds to zero or below.
    pub factor_underflow: &'static str,
    /// Field name of the accepted volumes in the finite check.
    pub accepted: &'static str,
    /// A nonzero proposed volume with a positive factor rounds to zero.
    pub accepted_underflow: &'static str,
    /// Outward-rounded donor totals exceed the available water.
    pub totals: &'static str,
}

/// Donor groups and the two endpoint groups of every link.
pub(crate) struct Donors<'a, R: Runtime<DType = DType>> {
    /// Groups over every node. Input `e` is link `e` at endpoint A, input `L + e` at endpoint B.
    pub tree: &'a TransferTree<R>,
    /// Group of endpoint A per link, I64 `[L]`.
    pub a: &'a Tensor<R>,
    /// Group of endpoint B per link, I64 `[L]`.
    pub b: &'a Tensor<R>,
    /// Group count that scales the assembly bound.
    pub groups: usize,
}

/// Accepted link volumes with their donor totals.
pub(crate) struct DonorLimit<R: Runtime<DType = DType>> {
    /// Accepted volume per link, positive from A to B, `[L]`.
    pub signed: Tensor<R>,
    /// Donor factor applied to each link, `[L]`.
    pub factor: Tensor<R>,
    /// Water each group receives.
    pub incoming: Tensor<R>,
    /// Water each group gives.
    pub outgoing: Tensor<R>,
    /// Largest aggregation rounding over every group, times the group count, in m³.
    pub aggregation_bound: f64,
}

/// Limits `proposed` volumes so no donor group gives more than `available`.
///
/// Each limited donor's factor is `(available - margin) / (demand + bound)`, rounded down once
/// more. The margin grows with the donor's link count. A positive volume draws on endpoint A.
pub(crate) fn limit<R: Runtime<DType = DType>>(
    c: &R::Client,
    donors: &Donors<'_, R>,
    proposed: &Tensor<R>,
    available: &Tensor<R>,
    zero: &Tensor<R>,
    one: &Tensor<R>,
    texts: &DonorTexts,
) -> Result<DonorLimit<R>, LinkError>
where
    R::Client: LinkOps<R>,
{
    let tiny = f64::from_bits(1);
    let (positive, negative) = split(c, proposed, zero)?;
    let water = AggregationKind::Water;
    let demands = donors
        .tree
        .aggregate(c, &c.cat(&[&positive, &negative], 0)?, water)?;
    let upper = water_transfers::upward_add(c, &demands.value, &demands.bound)?;
    let limited = c.gt(&upper, available)?;
    let margin = c.add_scalar(&c.mul(available, &donors.tree.degree_margin)?, tiny)?;
    let lower = c.sub(available, &margin)?;
    let has_water = c.gt(available, zero)?;
    let constrained = c.mul(&limited, &has_water)?;
    water_transfers::reject(
        c,
        &c.mul(&constrained, &c.le(&lower, zero)?)?,
        texts.unrepresentable,
    )?;
    let numerator = c.where_cond(&constrained, &lower, zero)?;
    let denominator = c.where_cond(&limited, &upper, one)?;
    let ratio = c.div(&numerator, &denominator)?;
    let rounding = c.add_scalar(&c.mul_scalar(&c.abs(&ratio)?, f64::EPSILON)?, tiny)?;
    let conservative = c.sub(&ratio, &rounding)?;
    water_transfers::reject(
        c,
        &c.mul(&constrained, &c.le(&conservative, zero)?)?,
        texts.factor_underflow,
    )?;
    let limited_factor = c.where_cond(&has_water, &conservative, zero)?;
    let group_factor = c.where_cond(&limited, &limited_factor, one)?;
    let factor_a = c.index_select(&group_factor, 0, donors.a)?;
    let factor_b = c.index_select(&group_factor, 0, donors.b)?;
    let factor = c.where_cond(&c.lt(proposed, zero)?, &factor_b, &factor_a)?;
    let signed = c.mul(proposed, &factor)?;
    water_transfers::finite(c, &signed, texts.accepted)?;
    water_transfers::reject(
        c,
        &c.mul(
            &c.mul(&c.gt(&c.abs(proposed)?, zero)?, &c.gt(&factor, zero)?)?,
            &c.eq(&signed, zero)?,
        )?,
        texts.accepted_underflow,
    )?;
    let (positive, negative) = split(c, &signed, zero)?;
    let outgoing = donors
        .tree
        .aggregate(c, &c.cat(&[&positive, &negative], 0)?, water)?;
    let incoming = donors
        .tree
        .aggregate(c, &c.cat(&[&negative, &positive], 0)?, water)?;
    let protected = water_transfers::upward_add(c, &outgoing.value, &outgoing.bound)?;
    water_transfers::reject(c, &c.gt(&protected, available)?, texts.totals)?;
    let aggregation_bound = assembly_bound(c, &incoming.bound, &outgoing.bound, donors.groups)?;
    Ok(DonorLimit {
        signed,
        factor,
        incoming: incoming.value,
        outgoing: outgoing.value,
        aggregation_bound,
    })
}

/// Positive parts and negated negative parts of `signed`.
pub(crate) fn split<R: Runtime<DType = DType>>(
    c: &R::Client,
    signed: &Tensor<R>,
    zero: &Tensor<R>,
) -> Result<(Tensor<R>, Tensor<R>), LinkError>
where
    R::Client: LinkOps<R>,
{
    Ok((
        c.where_cond(&c.gt(signed, zero)?, signed, zero)?,
        c.where_cond(&c.lt(signed, zero)?, &c.neg(signed)?, zero)?,
    ))
}

/// Largest summed incoming and outgoing rounding bound, times `groups`, outward rounded.
pub(crate) fn assembly_bound<R: Runtime<DType = DType>>(
    c: &R::Client,
    incoming: &Tensor<R>,
    outgoing: &Tensor<R>,
    groups: usize,
) -> Result<f64, LinkError>
where
    R::Client: LinkOps<R>,
{
    let combined = water_transfers::upward_add(c, incoming, outgoing)?;
    let largest = c.max(&combined, &[0], false)?.item::<f64>()?;
    if largest == 0.0 {
        Ok(0.0)
    } else {
        upward_mul(largest, groups as f64).map_err(|error| LinkError::Numerical(error.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::assembly_bound;
    use crate::numerics::water_transfers::{AggregationKind, TransferTree};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    use numr::tensor::Tensor;

    #[test]
    fn incoming_and_outgoing_assembly_errors_both_charge_closed_mass()
    -> Result<(), Box<dyn std::error::Error>> {
        let device = CpuDevice::new();
        let client = CpuClient::new(device.clone());
        let ids = ["in-large", "in-small", "out-large", "out-small"];
        let targets: Vec<_> = ids
            .iter()
            .enumerate()
            .flat_map(|(edge, id)| [(0, edge, *id), (1, ids.len() + edge, *id)])
            .collect();
        let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 2, 2 * ids.len(), &targets)?;
        let small = 3.0 * f64::EPSILON / 2.0;
        let endpoints = Tensor::<CpuRuntime>::from_slice(
            &[1.0, small, 0.0, 0.0, 0.0, 0.0, 1.0, small],
            &[8],
            &device,
        )?;
        let reverse = Tensor::<CpuRuntime>::from_slice(
            &[0.0, 0.0, 1.0, small, 1.0, small, 0.0, 0.0],
            &[8],
            &device,
        )?;
        let outgoing = tree.aggregate(&client, &endpoints, AggregationKind::Water)?;
        let incoming = tree.aggregate(&client, &reverse, AggregationKind::Water)?;
        let incoming_bounds = incoming.bound.try_to_vec::<f64>()?;
        let outgoing_bounds = outgoing.bound.try_to_vec::<f64>()?;
        assert!(incoming_bounds.iter().all(|value| *value > 0.0));
        assert!(outgoing_bounds.iter().all(|value| *value > 0.0));
        let exact_charge: f64 = incoming_bounds.iter().chain(&outgoing_bounds).sum();
        let largest_single_side = incoming_bounds
            .iter()
            .chain(&outgoing_bounds)
            .copied()
            .fold(0.0, f64::max);
        assert!(2.0 * largest_single_side < exact_charge);
        let charged = assembly_bound::<CpuRuntime>(&client, &incoming.bound, &outgoing.bound, 2)?;
        assert!(charged >= exact_charge);
        Ok(())
    }
}
