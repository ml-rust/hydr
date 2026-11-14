use super::{
    paired_sources::PairedSources, source_sonic_types::SonicSide, wave_curve_types::CurveValue,
    wet_wave_types::WetWaveDomain,
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult},
    section::{SupportedAreaQuery, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) struct WetFanMembership<R: Runtime<DType = DType>> {
    pub(super) proved: Tensor<R>,
    pub(super) excluded: Tensor<R>,
    pub(super) pending: Tensor<R>,
    pub(super) coincidence: Tensor<R>,
    pub(super) separation: CurveValue<R>,
    pub(super) lower_difference: solvr::common::arithmetic::CertifiedExpansion<R>,
    pub(super) upper_difference: solvr::common::arithmetic::CertifiedExpansion<R>,
}

/// Compare different equation roots through retained source-relative expressions, never interval overlap.
pub(super) fn evaluate<R>(
    a: &mut Arithmetic<'_, R>,
    sources: &PairedSources<'_, R>,
    wave: &WetWaveDomain<R>,
    rows: &Tensor<R>,
    sonic: &SupportedAreaQuery<R>,
    side: SonicSide,
) -> HydraulicResult<WetFanMembership<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let i = side.index();
    let parent = c.index_select(&wave.star.parent_rows, 0, rows)?;
    let selected = sources.select_rows(&parent, sonic)?;
    let star = wave.star.query.select(a, rows)?;
    let ds = source_difference(
        a,
        &selected.bands,
        &star,
        &selected.sources[i],
        &selected.boundaries,
    )?;
    let dp = source_difference(
        a,
        &selected.bands,
        sonic,
        &selected.sources[i],
        &selected.boundaries,
    )?;
    let lower = a.expansion_sub(&ds.lower, &dp.upper)?;
    let upper = a.expansion_sub(&ds.upper, &dp.lower)?;
    let low = a.expansion_interval(&lower)?;
    let high = a.expansion_interval(&upper)?;
    let zero = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
    let one = Tensor::full_scalar(rows.shape(), a.dtype(), 1., c.device())?;
    let ordered = c.le(low.lower(), high.upper())?;
    let physical = a.bounds(
        &c.where_cond(&ordered, low.lower(), &zero)?,
        &c.where_cond(&ordered, high.upper(), &zero)?,
    )?;
    let wave_side = &wave.sides[i];
    let rare = c.index_select(&wave_side.rarefaction_possible, 0, rows)?;
    let shock = c.index_select(&wave_side.shock_possible, 0, rows)?;
    let equality = c.index_select(&wave_side.equality_possible, 0, rows)?;
    let definite = c.mul(
        &rare,
        &c.mul(&c.eq(&shock, &zero)?, &c.eq(&equality, &zero)?)?,
    )?;
    let tail = crate::channel::flow::surveyed::equation_ops::select_interval(
        a,
        &wave_side.tail.physical,
        rows,
    )?;
    let tail_available = c.mul(
        &definite,
        &c.index_select(&wave_side.tail.available, 0, rows)?,
    )?;
    let (tail_proved, tail_excluded) = if side == SonicSide::Left {
        (c.ge(tail.lower(), &zero)?, c.lt(tail.upper(), &zero)?)
    } else {
        (c.le(tail.upper(), &zero)?, c.gt(tail.lower(), &zero)?)
    };
    let proved = c.maximum(
        &c.mul(&tail_available, &tail_proved)?,
        &c.mul(&ordered, &c.le(physical.upper(), &zero)?)?,
    )?;
    let excluded = c.maximum(
        &c.mul(&tail_available, &tail_excluded)?,
        &c.mul(&ordered, &c.gt(physical.lower(), &zero)?)?,
    )?;
    let conflict = c.mul(&proved, &excluded)?;
    let consistent = c.sub(&one, &conflict)?;
    let proved = c.mul(&proved, &consistent)?;
    let excluded = c.mul(&excluded, &consistent)?;
    let pending = c.sub(&one, &c.maximum(&proved, &excluded)?)?;
    let coincidence = c.mul(
        &ordered,
        &c.mul(
            &super::wet_star_endpoint::exact_zero(a, &lower)?,
            &super::wet_star_endpoint::exact_zero(a, &upper)?,
        )?,
    )?;
    let retained = a.expansion_from_interval(&physical)?;
    Ok(WetFanMembership {
        proved,
        excluded,
        pending,
        coincidence,
        lower_difference: lower,
        upper_difference: upper,
        separation: CurveValue {
            retained,
            physical,
            available: c.mul(&ordered, &consistent)?,
        },
    })
}
