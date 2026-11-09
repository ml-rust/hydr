//! Physical demand caps and independent original-potential reconciliation.

use crate::cpu::{
    CpuTensor,
    level_pool::{CpuPoolCap, CpuPoolContext, CpuPoolCounter},
};
use crate::storage::LevelPoolError;
use numr::ops::{BinaryOps, CompareOps, UnaryOps};

pub(super) fn cap(
    context: &CpuPoolContext,
    potential: &CpuTensor,
    available: &CpuTensor,
) -> Result<(CpuPoolCap, CpuTensor, CpuTensor), LevelPoolError> {
    let c = &context.client;
    let actual = c.minimum(potential, available)?;
    let (remaining, witness) = super::trial::exact_zero_add(context, available, &c.neg(&actual)?)?;
    context.reject(
        &c.mul(
            &c.gt(&actual, &context.zero)?,
            &c.eq(&remaining, available)?,
        )?,
        "positive actual source loss is absorbed. Rescale storage/source magnitudes or choose an interval with resolvable process volume",
    )?;
    let (unmet, unmet_witness) =
        super::trial::exact_zero_add(context, potential, &c.neg(&actual)?)?;
    let reconciled = c.add(&actual, &unmet)?;
    let residual = c.sub(&reconciled, potential)?;
    let consistency = c.add(&residual, &unmet_witness)?;
    let mut bound = super::bounds::rounding(context, &reconciled)?;
    for value in [&residual, &consistency] {
        bound = super::bounds::add(context, &bound, &super::bounds::rounding(context, value)?)?;
    }
    context.finite(&consistency, "potential-demand cap reconciliation")?;
    context.reject(
        &c.gt(&c.abs(&consistency)?, &bound)?,
        "potential-demand actual/unmet arithmetic reconciliation",
    )?;
    context.nonnegative(&remaining, "post-loss algebraic budget")?;
    context.nonnegative(&unmet, "unmet potential demand")?;
    Ok((
        CpuPoolCap {
            potential_volume_m3: potential.clone(),
            actual_volume_m3: actual,
            unmet_volume_m3: unmet,
            unmet_witness_m3: unmet_witness,
            reconciliation_residual_m3: residual,
            reconciliation_consistency_m3: consistency,
            reconciliation_bound_m3: bound,
            availability_limited: c.gt(potential, available)?,
        },
        remaining,
        witness,
    ))
}

pub(super) fn counter(
    context: &CpuPoolContext,
    before: &CpuTensor,
    bound: &CpuTensor,
    increment: &CpuTensor,
) -> Result<CpuPoolCounter, LevelPoolError> {
    let (after, witness) = super::trial::exact_zero_add(context, before, increment)?;
    let next_bound = super::bounds::accumulate(context, bound, &witness)?;
    Ok(CpuPoolCounter {
        before_m3: before.clone(),
        after_m3: after,
        witness_m3: witness,
        bound_before_m3: bound.clone(),
        bound_after_m3: next_bound,
    })
}
