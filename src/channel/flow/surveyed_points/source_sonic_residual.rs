use super::{
    paired_sources::PairedSources,
    potential_absolute::{admitted_analytic, endpoint_potential},
    potential_increment::endpoint_increment,
    potential_range::{PotentialContext, PotentialEvaluation, select_expansion},
    source_sonic_types::SonicSide,
    wave_curve_types::{CurveValue, scatter_value, unavailable},
    wet_star_endpoint::exact_zero,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar, select_interval},
        error::{HydraulicResult, checked},
    },
    section::{PhysicalEndpoint, SupportedAreaQuery, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn endpoint<R>(
    parent: &mut PairedSources<'_, R>,
    rows: &Tensor<R>,
    query: &SupportedAreaQuery<R>,
    endpoint: PhysicalEndpoint,
    side: SonicSide,
    epsilon: &CertifiedExpansion<R>,
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
    let n = rows.numel();
    let (target, raw, depth, sound, dry_rows, analytic_rows, local_rows, far_rows) =
        checked::<R, _>(c, "source sonic endpoint operands", |a| {
            let target = query.endpoint(a, &stage.bands, endpoint)?;
            let raw = target.retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
            let state = parent
                .geometry
                .recover_query(a, &stage.bands, &target)?
                .state;
            let gravity = scalar(a, parent.evaluation.config.gravity_m_per_s2)?;
            let sound = gravity_depth_root(a, &gravity, &state.hydraulic_depth, None)?;
            let sound = a.expansion_from_interval(&sound)?;
            let analytic = admitted_analytic(
                a,
                &stage.bands,
                &raw,
                &state.depth,
                parent.evaluation.config.gravity_m_per_s2,
            )?;
            let zero = Tensor::zeros(&[n], a.dtype(), c.device())?;
            let dry = exact_zero(a, &raw)?;
            let epsilon_bounds = a.expansion_interval(epsilon)?;
            let analytic_mask = c.mul(
                &c.eq(&dry, &zero)?,
                &c.mul(&analytic.eligible, &c.gt(epsilon_bounds.lower(), &zero)?)?,
            )?;
            let remaining = c.eq(&c.maximum(&dry, &analytic_mask)?, &zero)?;
            let delta = source_difference(
                a,
                &stage.bands,
                &target,
                &stage.sources[side.index()],
                &stage.boundaries,
            )?
            .lower;
            let conditioner = a.expansion_add(&raw, &delta)?;
            let conditioner = a.expansion_interval(&conditioner)?;
            let far = c.mul(&remaining, &c.lt(conditioner.upper(), &zero)?)?;
            let local = c.mul(&remaining, &c.eq(&far, &zero)?)?;
            Ok((
                target,
                raw,
                state.depth,
                sound,
                c.nonzero(&dry)?.flatten()?,
                c.nonzero(&analytic_mask)?.flatten()?,
                c.nonzero(&local)?.flatten()?,
                c.nonzero(&far)?.flatten()?,
            ))
        })?;
    let mut result = checked(c, "source sonic unavailable endpoint", |a| {
        unavailable(a, n)
    })?;
    if dry_rows.numel() != 0 {
        result = checked(c, "source sonic exact dry residual", |a| {
            let epsilon = select_expansion(a, epsilon, &dry_rows)?;
            let zero =
                a.expansion_point(&Tensor::zeros(dry_rows.shape(), a.dtype(), c.device())?)?;
            let retained = a.expansion_sub(&zero, &epsilon)?;
            let value = CurveValue {
                physical: a.expansion_interval(&retained)?,
                retained,
                available: Tensor::ones(dry_rows.shape(), a.dtype(), c.device())?,
            };
            scatter_value(a, result, &dry_rows, &value)
        })?;
    }
    if analytic_rows.numel() != 0 {
        result = checked(c, "source sonic authenticated analytic residual", |a| {
            let bands = stage.bands.select(a, &analytic_rows)?;
            let raw = select_expansion(a, &raw, &analytic_rows)?;
            let depth = select_interval(a, &depth, &analytic_rows)?;
            let target = admitted_analytic(
                a,
                &bands,
                &raw,
                &depth,
                parent.evaluation.config.gravity_m_per_s2,
            )?;
            let epsilon = select_expansion(a, epsilon, &analytic_rows)?;
            let value = super::potential_sonic::residual(a, &target, &epsilon)?;
            scatter_value(
                a,
                result,
                &analytic_rows,
                &CurveValue {
                    retained: value.retained_residual,
                    physical: value.physical_residual,
                    available: value.eligible,
                },
            )
        })?;
    }
    for (selected_rows, far) in [(&local_rows, false), (&far_rows, true)] {
        if selected_rows.numel() == 0 {
            continue;
        }
        let (bands, source, target, boundaries) =
            checked(c, "source sonic endpoint branch selection", |a| {
                Ok((
                    stage.bands.select(a, selected_rows)?,
                    stage.sources[side.index()].select(a, selected_rows)?,
                    target.select(a, selected_rows)?,
                    stage.boundaries.select(a, selected_rows)?,
                ))
            })?;
        let context = PotentialContext::new(
            parent.geometry,
            &bands,
            &source,
            &target,
            &boundaries,
            PotentialEvaluation {
                config: parent.evaluation.config,
                panels: parent.evaluation.panels,
                work: parent.evaluation.work,
            },
        )?;
        let term = if far {
            endpoint_potential(&context, PhysicalEndpoint::Lower)?
        } else {
            endpoint_increment(&context, PhysicalEndpoint::Lower)?
        };
        result = checked(c, "source sonic cancellation before target terms", |a| {
            let sound = select_expansion(a, &sound, selected_rows)?;
            let retained = if far {
                let epsilon = select_expansion(a, epsilon, selected_rows)?;
                let zero = a.expansion_point(&Tensor::zeros(
                    selected_rows.shape(),
                    a.dtype(),
                    c.device(),
                )?)?;
                let invariant = a.expansion_sub(&zero, &epsilon)?;
                let partial = a.expansion_add(&invariant, &term.retained)?;
                a.expansion_add(&partial, &sound)?
            } else {
                let velocity = if side == SonicSide::Left {
                    &stage.velocity.left
                } else {
                    &stage.velocity.right
                };
                let velocity = select_expansion(a, velocity, selected_rows)?;
                let partial = if side == SonicSide::Left {
                    a.expansion_sub(&sound, &velocity)?
                } else {
                    a.expansion_add(&sound, &velocity)?
                };
                a.expansion_add(&partial, &term.retained)?
            };
            let value = CurveValue {
                physical: a.expansion_interval(&retained)?,
                retained,
                available: Tensor::ones(selected_rows.shape(), a.dtype(), c.device())?,
            };
            scatter_value(a, result, selected_rows, &value)
        })?;
    }
    Ok(result)
}
