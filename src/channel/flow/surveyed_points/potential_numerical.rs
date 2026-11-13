use super::potential_range::{
    PotentialContext, PotentialValue, from_interval, scatter, zero_value,
};
use super::{analytic_endpoint, potential_analytic, potential_callbacks};
use crate::channel::{
    flow::surveyed::{
        derivatives::WetDepthPotential,
        equation_ops::{scalar, scale},
        error::{HydraulicResult, checked},
        lane_quadrature::numerical_contributions,
    },
    section::ConditionalBandSegments,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps, UnaryOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::{CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions};

/// Selected faces retain all physical bands in their original order, including zero lanes.
pub(super) fn wet_contributions<R>(
    context: &PotentialContext<'_, R>,
    segments: &ConditionalBandSegments<R>,
    faces: &Tensor<R>,
) -> HydraulicResult<PotentialValue<R>>
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
    let band_count = context.bands.band_count;
    let selected = checked(c, "point wet potential face selection", |a| {
        let n = context.bands.section_owner.numel();
        let mask = Tensor::zeros(&[n], DType::F64, c.device())?;
        let mask = c.index_put(
            &mask,
            0,
            faces,
            &Tensor::ones(&[faces.numel()], DType::F64, c.device())?,
        )?;
        let repeated = c.mul(
            &mask.unsqueeze(1)?,
            &Tensor::ones(&[n, band_count], DType::F64, c.device())?,
        )?;
        let rows = c
            .nonzero(&repeated.reshape(&[n * band_count])?)?
            .flatten()?;
        segments.select(a, &rows)
    })?;
    let count = selected.owners().numel();
    let (mut values, ordinary, wide) = checked(c, "point wet potential lane dispatch", |a| {
        let rows = c.arange(0.0, count as f64, 1.0, DType::I64)?;
        let endpoints = selected.clipped_endpoints(a, &rows)?;
        let source = analytic_endpoint::evaluate(
            a,
            &endpoints,
            analytic_endpoint::EndpointSide::Source,
            context.evaluation.config.gravity_m_per_s2,
        )?;
        let target = analytic_endpoint::evaluate(
            a,
            &endpoints,
            analytic_endpoint::EndpointSide::Target,
            context.evaluation.config.gravity_m_per_s2,
        )?;
        let analytic = potential_analytic::difference(
            a,
            &source,
            &target,
            endpoints.source().top_width(),
            endpoints.target().top_width(),
            selected.retained_gap(),
            context.evaluation.config.gravity_m_per_s2,
        )?;
        let zero = Tensor::zeros(&[count], DType::F64, c.device())?;
        let active = c.mul(
            selected.active_mask(),
            &c.eq(selected.zero_gap_mask(), &zero)?,
        )?;
        let analytic_rows = c.nonzero(&c.mul(&active, &analytic.eligible)?)?.flatten()?;
        let mut values = zero_value(a, count)?;
        if analytic_rows.numel() != 0 {
            let retained = super::potential_range::select_expansion(
                a,
                &analytic.retained_increment,
                &analytic_rows,
            )?;
            let physical = crate::channel::flow::surveyed::equation_ops::select_interval(
                a,
                &analytic.physical_increment,
                &analytic_rows,
            )?;
            values = scatter(
                a,
                values,
                &analytic_rows,
                &PotentialValue {
                    retained,
                    physical,
                    analytic: None,
                },
            )?;
        }
        let remaining = c.mul(&active, &c.eq(&analytic.eligible, &zero)?)?;
        let first = endpoints.source().physical_area();
        let second = endpoints.target().physical_area();
        let lower = c.minimum(first.lower(), second.lower())?;
        let upper = c.maximum(first.upper(), second.upper())?;
        let upper = a.point(&upper)?;
        // Half the upper endpoint avoids overflow in the equivalent upper <= 2*lower test.
        let half_upper = scale(a, &upper, 0.5)?;
        let conditioned = c.mul(&c.gt(&lower, &zero)?, &c.le(half_upper.upper(), &lower)?)?;
        let ordinary = c.nonzero(&c.mul(&remaining, &conditioned)?)?.flatten()?;
        let wide = c
            .nonzero(&c.mul(&remaining, &c.eq(&conditioned, &zero)?)?)?
            .flatten()?;
        Ok((values, ordinary, wide))
    })?;
    for (rows, ordinary_map) in [(&ordinary, true), (&wide, false)] {
        if rows.numel() == 0 {
            continue;
        }
        let contribution = integrate_selected(context, &selected, rows, ordinary_map)?;
        values = checked(c, "point wet potential contribution assembly", |a| {
            scatter(a, values, rows, &contribution)
        })?;
    }
    checked(c, "point potential ordered band sum", |a| {
        let n = faces.numel();
        let shape = [n, band_count];
        let high = values.retained.high().reshape(&shape)?;
        let low = values.retained.low().reshape(&shape)?;
        let error = values.retained.abs_error().reshape(&shape)?;
        let lower = values.physical.lower().reshape(&shape)?;
        let upper = values.physical.upper().reshape(&shape)?;
        let mut total = zero_value(a, n)?;
        for band in 0..band_count {
            let column = |v: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(v.narrow(1, band, 1)?.squeeze(Some(1)))
            };
            let term = a.expansion_bounds(&column(&high)?, &column(&low)?, &column(&error)?)?;
            total.retained = a.expansion_add(&total.retained, &term)?;
            let term = a.bounds(&column(&lower)?, &column(&upper)?)?;
            total.physical = a.add(&total.physical, &term)?;
        }
        Ok(total)
    })
}

fn integrate_selected<R>(
    context: &PotentialContext<'_, R>,
    segments: &ConditionalBandSegments<R>,
    rows: &Tensor<R>,
    ordinary: bool,
) -> HydraulicResult<PotentialValue<R>>
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
    let selected = checked(c, "point numerical potential selection", |a| {
        segments.select(a, rows)
    })?;
    let depth = if ordinary {
        None
    } else {
        Some(checked(c, "point shifted wet potential geometry", |a| {
            let owner = c.arange(0.0, rows.numel() as f64, 1.0, DType::I64)?;
            selected.wet_depth_geometry(a, &owner)
        })?)
    };
    let left = Tensor::zeros(&[rows.numel()], DType::F64, c.device())?;
    let right = Tensor::ones(&[rows.numel()], DType::F64, c.device())?;
    let gravity = context.evaluation.config.gravity_m_per_s2;
    let result = numerical_contributions(
        c,
        &left,
        &right,
        CertifiedSimpsonOptions {
            panels: context.evaluation.panels,
            max_points: context.evaluation.config.max_native_query_points,
        },
        context.evaluation.work,
        |a, owner, parameter| {
            if let Some(depth) = &depth {
                WetDepthPotential::from_geometry(a, depth, owner, parameter, gravity)?.value(a)
            } else {
                let gravity = scalar(a, gravity)?;
                potential_callbacks::integrand(a, &selected, owner, parameter, &gravity)
            }
        },
        |a, owner, parameter| {
            if let Some(depth) = &depth {
                let derivative =
                    WetDepthPotential::from_geometry(a, depth, owner, parameter, gravity)?
                        .fourth_derivative(a)?;
                Ok(c.maximum(&c.abs(derivative.lower())?, &c.abs(derivative.upper())?)?)
            } else {
                let gravity = scalar(a, gravity)?;
                potential_callbacks::fourth_bound(a, &selected, owner, parameter, &gravity)
            }
        },
    )?;
    checked(c, "point numerical potential certificate", |a| {
        let physical = a.bounds(result.lower(), result.upper())?;
        let raw_bounds = a.bounds(physical.lower(), physical.upper())?;
        let mut value = from_interval(a, raw_bounds)?;
        let gap = selected.physical_gap();
        let zero = Tensor::zeros(gap.lower().shape(), DType::F64, c.device())?;
        let physical = a.bounds(
            &c.where_cond(
                &c.ge(gap.lower(), &zero)?,
                &c.maximum(physical.lower(), &zero)?,
                physical.lower(),
            )?,
            &c.where_cond(
                &c.le(gap.upper(), &zero)?,
                &c.minimum(physical.upper(), &zero)?,
                physical.upper(),
            )?,
        )?;
        value.physical = physical;
        Ok(value)
    })
}
