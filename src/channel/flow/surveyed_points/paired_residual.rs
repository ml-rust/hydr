use super::paired_sources::PairedSources;
use super::potential_absolute::endpoint_potential;
use super::potential_range::{PotentialContext, PotentialEvaluation};
use super::wave_curve;
use super::wave_curve_types::{CurveValue, scatter_value, unavailable};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{PhysicalEndpoint, SupportedAreaQuery, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

pub(super) fn endpoint<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let stage = parent.select_rows(rows, query)?;
    let c = parent.geometry.client();
    let n = stage.original_rows.numel();
    let (target_query, far, local_rows, far_rows, union_rows) =
        checked(c, "paired residual branch classification", |a| {
            let target = query.endpoint(a, &stage.bands, endpoint)?;
            let area = target.retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            let mut far = [zero.clone(), zero.clone()];
            let mut local_rows = [
                Tensor::zeros(&[0], DType::I64, c.device())?,
                Tensor::zeros(&[0], DType::I64, c.device())?,
            ];
            let mut far_rows = [local_rows[0].clone(), local_rows[1].clone()];
            for (side, source) in stage.sources.iter().enumerate() {
                let delta =
                    source_difference(a, &stage.bands, &target, source, &stage.boundaries)?.lower;
                let class = wave_curve::classify(a, source.dry_mask(), &delta)?;
                let conditioner = a.expansion_add(&area, &delta)?;
                let conditioner = a.expansion_interval(&conditioner)?;
                far[side] = c.mul(
                    &stage.velocity.both_wet,
                    &c.mul(&class.rarefaction, &c.lt(conditioner.upper(), &zero)?)?,
                )?;
                local_rows[side] = c
                    .nonzero(&c.mul(&stage.velocity.both_wet, &c.eq(&far[side], &zero)?)?)?
                    .flatten()?;
                far_rows[side] = c.nonzero(&far[side])?.flatten()?;
            }
            let union = c.nonzero(&c.maximum(&far[0], &far[1])?)?.flatten()?;
            Ok((target, far, local_rows, far_rows, union))
        })?;
    let mut local = [
        checked(c, "paired left local placeholder", |a| unavailable(a, n))?,
        checked(c, "paired right local placeholder", |a| unavailable(a, n))?,
    ];
    for (side, rows) in local_rows.iter().enumerate() {
        if rows.numel() == 0 {
            continue;
        }
        let (bands, source, query, boundaries) = checked(c, "paired local wave selection", |a| {
            Ok((
                stage.bands.select(a, rows)?,
                stage.sources[side].select(a, rows)?,
                target_query.select(a, rows)?,
                stage.boundaries.select(a, rows)?,
            ))
        })?;
        let context = PotentialContext::new(
            parent.geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: parent.evaluation.config,
                panels: parent.evaluation.panels,
                work: parent.evaluation.work,
            },
        )?;
        let value = wave_curve::endpoint(&context, PhysicalEndpoint::Lower)?.curve;
        let dense = checked(c, "paired local wave assembly", |a| {
            let empty = unavailable(a, n)?;
            scatter_value(a, empty, rows, &value)
        })?;
        local[side] = dense;
    }
    let global_far = checked::<R, _>(c, "paired source cache parent rows", |a| {
        Ok([
            a.client().index_select(&stage.rows, 0, &far_rows[0])?,
            a.client().index_select(&stage.rows, 0, &far_rows[1])?,
        ])
    })?;
    let source_values = parent.source_values([&global_far[0], &global_far[1]])?;
    let mut far_sources = [
        checked(c, "paired left source placeholder", |a| unavailable(a, n))?,
        checked(c, "paired right source placeholder", |a| unavailable(a, n))?,
    ];
    for (side, value) in source_values.iter().enumerate() {
        if far_rows[side].numel() == 0 {
            continue;
        }
        far_sources[side] = checked(c, "paired far source assembly", |a| {
            let empty = unavailable(a, n)?;
            scatter_value(a, empty, &far_rows[side], value)
        })?;
    }
    let mut target_value = checked(c, "paired target placeholder", |a| unavailable(a, n))?;
    if union_rows.numel() != 0 {
        let (bands, source, query, boundaries) =
            checked(c, "paired shared target selection", |a| {
                Ok((
                    stage.bands.select(a, &union_rows)?,
                    stage.sources[0].select(a, &union_rows)?,
                    target_query.select(a, &union_rows)?,
                    stage.boundaries.select(a, &union_rows)?,
                ))
            })?;
        let context = PotentialContext::new(
            parent.geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: parent.evaluation.config,
                panels: parent.evaluation.panels,
                work: parent.evaluation.work,
            },
        )?;
        let value = endpoint_potential(&context, PhysicalEndpoint::Lower)?;
        target_value = checked(c, "paired shared target assembly", |a| {
            scatter_value(
                a,
                target_value,
                &union_rows,
                &CurveValue {
                    retained: value.retained,
                    physical: value.physical,
                    available: Tensor::ones(union_rows.shape(), a.dtype(), c.device())?,
                },
            )
        })?;
    }
    checked(c, "paired cancellation before target potentials", |a| {
        let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
        let retained_du = &stage.velocity.difference;
        let mut retained = a.expansion_bounds(
            &c.where_cond(&stage.velocity.both_wet, retained_du.high(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.low(), &zero)?,
            &c.where_cond(&stage.velocity.both_wet, retained_du.abs_error(), &zero)?,
        )?;
        for value in &far_sources {
            retained = a.expansion_sub(&retained, &value.retained)?;
        }
        for value in &local {
            retained = a.expansion_add(&retained, &value.retained)?;
        }
        for mask in &far {
            let term = a.expansion_bounds(
                &c.where_cond(mask, target_value.retained.high(), &zero)?,
                &c.where_cond(mask, target_value.retained.low(), &zero)?,
                &c.where_cond(mask, target_value.retained.abs_error(), &zero)?,
            )?;
            retained = a.expansion_add(&retained, &term)?;
        }
        let physical = a.expansion_interval(&retained)?;
        Ok(CurveValue {
            retained,
            physical,
            available: stage.velocity.both_wet,
        })
    })
}
