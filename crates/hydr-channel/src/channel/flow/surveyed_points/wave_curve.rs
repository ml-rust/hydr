use super::potential_increment::endpoint_increment;
use super::potential_range::{PotentialContext, PotentialEvaluation, select_expansion};
use super::shock_points;
#[cfg(test)]
use super::wave_curve_types::CurveRange;
use super::wave_curve_types::{
    CurveClassification, CurveValue, EndpointCurve, empty_offset, scatter_offset, scatter_value,
    unavailable,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar, select_interval},
        error::{Arithmetic, HydraulicResult, checked},
    },
    section::{PhysicalEndpoint, pressure_endpoint, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn endpoint<R>(
    context: &PotentialContext<'_, R>,
    endpoint: PhysicalEndpoint,
) -> HydraulicResult<EndpointCurve<R>>
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
    let c = context.geometry.client();
    let n = context.bands.section_owner.numel();
    let (query, delta, mut result) = checked(c, "point side curve classification", |a| {
        let query = context.query.endpoint(a, context.bands, endpoint)?;
        let delta =
            source_difference(a, context.bands, &query, context.source, context.boundaries)?.lower;
        let classification = classify(a, context.source.dry_mask(), &delta)?;
        let result = EndpointCurve {
            curve: unavailable(a, n)?,
            classification,
            actual_shock: empty_offset(a, n)?,
            conditional_shock: empty_offset(a, n)?,
            equality_limit: unavailable(a, n)?,
        };
        Ok((query, delta, result))
    })?;
    let equal_rows = c.nonzero(&result.classification.equality)?.flatten()?;
    if equal_rows.numel() != 0 {
        let (curve, limit) = checked(c, "point equality curve and sound limit", |a| {
            let hydraulic_depth =
                select_interval(a, context.source.hydraulic_depth_m(), &equal_rows)?;
            let gravity = scalar(a, context.evaluation.config.gravity_m_per_s2)?;
            let sound = gravity_depth_root(a, &gravity, &hydraulic_depth, None)?;
            let retained = a.expansion_from_interval(&sound)?;
            let one = Tensor::ones(&[equal_rows.numel()], a.dtype(), c.device())?;
            let mut curve = unavailable(a, equal_rows.numel())?;
            curve.available = one.clone();
            Ok((
                curve,
                CurveValue {
                    retained,
                    physical: sound,
                    available: one,
                },
            ))
        })?;
        result.curve = checked(c, "point equality curve assembly", |a| {
            scatter_value(a, result.curve, &equal_rows, &curve)
        })?;
        result.equality_limit = checked(c, "point equality limit assembly", |a| {
            scatter_value(a, result.equality_limit, &equal_rows, &limit)
        })?;
    }
    let (rare_rows, shock_rows, conditional_rows) =
        checked(c, "point side curve branch rows", |a| {
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            let bounds = a.expansion_interval(&delta)?;
            let possible_rare = c.mul(
                &result.classification.uncertain,
                &c.lt(bounds.lower(), &zero)?,
            )?;
            let possible_shock = c.mul(
                &result.classification.uncertain,
                &c.gt(bounds.upper(), &zero)?,
            )?;
            Ok((
                c.nonzero(&c.maximum(&result.classification.rarefaction, &possible_rare)?)?
                    .flatten()?,
                c.nonzero(&result.classification.shock)?.flatten()?,
                c.nonzero(&possible_shock)?.flatten()?,
            ))
        })?;
    let mut rare = checked(c, "point unavailable rare branch", |a| unavailable(a, n))?;
    if rare_rows.numel() != 0 {
        let (bands, source, query, boundaries) = checked(c, "point rare curve selection", |a| {
            Ok((
                context.bands.select(a, &rare_rows)?,
                context.source.select(a, &rare_rows)?,
                query.select(a, &rare_rows)?,
                context.boundaries.select(a, &rare_rows)?,
            ))
        })?;
        let parents = match &context.source_rows {
            Some(parents) => Some(parents.select(c, &rare_rows)?),
            None => None,
        };
        let selected = PotentialContext::new(
            context.geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: context.evaluation.config,
                panels: context.evaluation.panels,
                work: context.evaluation.work,
            },
        )?
        .with_source_rows(parents);
        let increment = endpoint_increment(&selected, PhysicalEndpoint::Lower)?;
        rare = checked(c, "point rare branch capture", |a| {
            let value = CurveValue {
                retained: increment.retained,
                physical: increment.physical,
                available: Tensor::ones(&[rare_rows.numel()], a.dtype(), c.device())?,
            };
            scatter_value(a, rare, &rare_rows, &value)
        })?;
        let rows = c.nonzero(&result.classification.rarefaction)?.flatten()?;
        if rows.numel() != 0 {
            let value = checked(c, "point definite rare curve", |a| {
                let mut value = select_value(a, &rare, &rows)?;
                let zero = Tensor::zeros(value.physical.lower().shape(), a.dtype(), c.device())?;
                value.physical = a.bounds(
                    value.physical.lower(),
                    &c.minimum(value.physical.upper(), &zero)?,
                )?;
                Ok(value)
            })?;
            result.curve = checked(c, "point rare curve assembly", |a| {
                scatter_value(a, result.curve, &rows, &value)
            })?;
        }
    }
    let mut shock = checked(c, "point unavailable conditional shock", |a| {
        unavailable(a, n)
    })?;
    for (rows, conditional) in [(&shock_rows, false), (&conditional_rows, true)] {
        if rows.numel() == 0 {
            continue;
        }
        let (value, offset) = checked(c, "point shock curve evaluation", |a| {
            let bands = context.bands.select(a, rows)?;
            let source = context.source.select(a, rows)?;
            let query = query.select(a, rows)?;
            let boundaries = context.boundaries.select(a, rows)?;
            let target_area = query.retained_endpoint_area(a, &bands, &boundaries)?;
            let pressure = pressure_endpoint(
                a,
                context.geometry,
                &bands,
                &query,
                &source,
                &boundaries,
                context.evaluation.config.gravity_m_per_s2,
            )?;
            if conditional {
                shock_points::conditional(a, source.requested_areas_m2(), &target_area, &pressure)
            } else {
                shock_points::actual(a, source.requested_areas_m2(), &target_area, &pressure)
            }
        })?;
        if conditional {
            shock = checked(c, "point conditional shock curve capture", |a| {
                scatter_value(a, shock, rows, &value)
            })?;
            result.conditional_shock =
                checked(c, "point conditional shock domain assembly", |a| {
                    scatter_offset(a, result.conditional_shock, rows, &offset)
                })?;
        } else {
            result.curve = checked(c, "point actual shock curve assembly", |a| {
                scatter_value(a, result.curve, rows, &value)
            })?;
            result.actual_shock = checked(c, "point actual shock domain assembly", |a| {
                scatter_offset(a, result.actual_shock, rows, &offset)
            })?;
        }
    }
    let rows = c.nonzero(&result.classification.uncertain)?.flatten()?;
    if rows.numel() != 0 {
        let value = checked(c, "point conditional side curve envelope", |a| {
            let rare = select_interval(a, &rare.physical, &rows)?;
            let shock = select_interval(a, &shock.physical, &rows)?;
            let zero = Tensor::zeros(&[rows.numel()], a.dtype(), c.device())?;
            // Equality zero joins possible branches without asserting an equality identity.
            let lower = c.minimum(rare.lower(), &zero)?;
            let upper = c.maximum(shock.upper(), &zero)?;
            let physical = a.bounds(&lower, &upper)?;
            Ok(CurveValue {
                retained: a.expansion_from_interval(&physical)?,
                physical,
                available: Tensor::ones(&[rows.numel()], a.dtype(), c.device())?,
            })
        })?;
        result.curve = checked(c, "point conditional side curve assembly", |a| {
            scatter_value(a, result.curve, &rows, &value)
        })?;
    }
    Ok(result)
}

#[cfg(test)]
pub(super) fn range<R>(context: &PotentialContext<'_, R>) -> HydraulicResult<CurveRange<R>>
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
    let lower = endpoint(context, PhysicalEndpoint::Lower)?;
    let upper = endpoint(context, PhysicalEndpoint::Upper)?;
    let curve = checked::<R, _>(
        context.geometry.client(),
        "point monotone side curve range",
        |a| {
            let available = a
                .client()
                .mul(&lower.curve.available, &upper.curve.available)?;
            let physical = a.bounds(lower.curve.physical.lower(), upper.curve.physical.upper())?;
            Ok(CurveValue {
                retained: a.expansion_from_interval(&physical)?,
                physical,
                available,
            })
        },
    )?;
    Ok(CurveRange {
        lower,
        upper,
        curve,
    })
}

pub(super) fn classify<R>(
    a: &mut Arithmetic<'_, R>,
    dry: &Tensor<R>,
    delta: &CertifiedExpansion<R>,
) -> HydraulicResult<CurveClassification<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(dry.shape(), a.dtype(), c.device())?;
    let wet = c.eq(dry, &zero)?;
    let bounds = a.expansion_interval(delta)?;
    let shock = c.mul(&wet, &c.gt(bounds.lower(), &zero)?)?;
    let rarefaction = c.mul(&wet, &c.lt(bounds.upper(), &zero)?)?;
    let equality = c.mul(
        &wet,
        &c.mul(
            &c.eq(delta.high(), &zero)?,
            &c.mul(&c.eq(delta.low(), &zero)?, &c.eq(delta.abs_error(), &zero)?)?,
        )?,
    )?;
    let known = c.maximum(&equality, &c.maximum(&shock, &rarefaction)?)?;
    let uncertain = c.mul(&wet, &c.eq(&known, &zero)?)?;
    Ok(CurveClassification {
        shock,
        rarefaction,
        equality,
        uncertain,
        #[cfg(test)]
        dry_source: dry.clone(),
    })
}

fn select_value<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CurveValue<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CurveValue<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    Ok(CurveValue {
        retained: select_expansion(a, &value.retained, rows)?,
        physical: select_interval(a, &value.physical, rows)?,
        available: a.client().index_select(&value.available, 0, rows)?,
    })
}
