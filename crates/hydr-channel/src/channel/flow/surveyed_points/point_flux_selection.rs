use super::{
    paired_sources::PairedSources,
    point_flux_operands::{Operands, copy_tensor},
    point_flux_pressure::selected as selected_pressure,
    point_flux_projection::{center, choose},
    point_flux_types::{PointFluxKind, SelectedPointFlux},
    source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicResult, checked},
    section::{AreaAnchor, AreaDirection, AreaQuery, PhysicalEndpoint, source_difference},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::Interval, integrate::CertifiedSimpsonAlgorithms};

pub(super) fn evaluate<R>(
    sources: &mut PairedSources<'_, R>,
    operand: Operands<R>,
) -> HydraulicResult<SelectedPointFlux<R>>
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
    let c = sources.geometry.client();
    if operand.kind == PointFluxKind::Dry {
        return dry(sources, operand);
    }
    let (query, area, area_evaluation, area_discrepancy, area_projection) =
        checked::<R, _>(c, "coherent point area selection", |a| {
            let stage = sources.select_rows(&operand.parent, &operand.query)?;
            let midpoint = c.midpoint(
                operand.query.query.parameter.lower(),
                operand.query.query.parameter.upper(),
            )?;
            let endpoint = operand
                .query
                .with_parameter_point(a, &stage.bands, &midpoint)?;
            let mut raw = endpoint.retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
            if let Some([captured, _]) = &operand.captured {
                raw = a.expansion_bounds(captured.high(), captured.low(), captured.abs_error())?;
            } else if operand.kind == PointFluxKind::Star {
                let source = &stage.velocity.input().left_area;
                raw = a.expansion_bounds(
                    &c.where_cond(&stage.velocity.uniform, source.high(), raw.high())?,
                    &c.where_cond(&stage.velocity.uniform, source.low(), raw.low())?,
                    &c.where_cond(&stage.velocity.uniform, source.abs_error(), raw.abs_error())?,
                )?;
            }
            let area = center(a, &raw)?;
            let bounds = a.expansion_interval(&area)?;
            let zero = Tensor::zeros(bounds.lower().shape(), a.dtype(), c.device())?;
            super::point_flux_admission::positive(
                a,
                &operand.query,
                &c.le(bounds.lower(), &zero)?,
                "positive area",
            )?;
            let one = a.point(&Tensor::ones(
                bounds.lower().shape(),
                a.dtype(),
                c.device(),
            )?)?;
            let rebuilt = AreaQuery::band_affine(
                a,
                &stage.bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &area,
                &one,
            )?;
            c.copy_into(&rebuilt.original_rows, &operand.query.query.original_rows)?;
            let query = rebuilt.prove_support(a, &stage.bands)?;
            let selected = a.expansion_interval(&area)?;
            let area_discrepancy = a.sub(&operand.query.area, &selected)?;
            let projected = a.expansion_interval(&raw)?;
            let area_projection = a.sub(&projected, &selected)?;
            Ok((query, area, raw, area_discrepancy, area_projection))
        })?;
    let side = match operand.kind {
        PointFluxKind::Exterior(side) | PointFluxKind::Fan(side) => side,
        _ => SonicSide::Left,
    };
    let (discharge_evaluation, discharge_invariant_velocity, captured_identity, star_discharge) =
        if let Some([_, q]) = &operand.captured {
            checked::<R, _>(c, "captured point discharge", |a| {
                let stage = sources.select_rows(&operand.parent, &query)?;
                let velocity = if side == SonicSide::Left {
                    &stage.velocity.left
                } else {
                    &stage.velocity.right
                };
                Ok((
                    a.expansion_bounds(q.high(), q.low(), q.abs_error())?,
                    a.expansion_bounds(velocity.high(), velocity.low(), velocity.abs_error())?,
                    Tensor::ones(q.high().shape(), a.dtype(), c.device())?,
                    None,
                ))
            })?
        } else if let PointFluxKind::Fan(side) = operand.kind {
            let numerical = operand.numerical.as_ref().ok_or_else(|| {
                super::point_root_session::invalid("selected fan numerical evidence is absent")
            })?;
            numerical.check_rows(sources, side, &operand.owners, &operand.parent)?;
            let (discharge, velocity) =
                fan_discharge(sources, &operand.parent, &query, side, numerical)?;
            let identity =
                Tensor::zeros(operand.parent.shape(), discharge.high().dtype(), c.device())?;
            (discharge, velocity, identity, None)
        } else {
            let level = operand.star_level.as_ref().ok_or_else(|| {
                super::point_root_session::invalid("selected star panel level is absent")
            })?;
            let endpoint = super::wet_star_endpoint::evaluate(
                sources,
                &operand.parent,
                &query,
                PhysicalEndpoint::Lower,
                level,
            )?;
            checked::<R, _>(c, "uniform point captured discharge identity", |a| {
                let stage = sources.select_rows(&operand.parent, &query)?;
                let source = &stage.velocity.input().left_discharge;
                let q = &endpoint.sides[0].discharge;
                let discharge = a.expansion_bounds(
                    &c.where_cond(&stage.velocity.uniform, source.high(), q.high())?,
                    &c.where_cond(&stage.velocity.uniform, source.low(), q.low())?,
                    &c.where_cond(&stage.velocity.uniform, source.abs_error(), q.abs_error())?,
                )?;
                let velocity = &endpoint.sides[0].velocity;
                // The candidate holds both sides' root-bracket discharges intersected. It encloses Q*.
                let star = a.bounds(operand.discharge.lower(), operand.discharge.upper())?;
                Ok((
                    discharge,
                    a.expansion_bounds(velocity.high(), velocity.low(), velocity.abs_error())?,
                    copy_tensor(a, &stage.velocity.uniform)?,
                    Some(star),
                ))
            })?
        };
    checked::<R, _>(c, "coherent point momentum assembly", |a| {
        let stage = sources.select_rows(&operand.parent, &query)?;
        let state = sources
            .geometry
            .recover_query(a, &stage.bands, &query)?
            .state;
        let zero = Tensor::zeros(state.depth.lower().shape(), a.dtype(), c.device())?;
        super::point_flux_admission::positive(
            a,
            &query,
            &c.le(state.depth.lower(), &zero)?,
            "wet depth",
        )?;
        let anchored = a.expansion_interval(&discharge_evaluation)?;
        let invariant = a.expansion_interval(&discharge_invariant_velocity)?;
        let area_bounds = a.expansion_interval(&area)?;
        let discharge_product = a.mul(&area_bounds, &invariant)?;
        let (intersection, valid) =
            super::wet_star::intersection(a, &anchored, &discharge_product)?;
        let invalid = c.mul(&c.eq(&valid, &zero)?, &c.eq(&captured_identity, &zero)?)?;
        if c.nonzero(&invalid)?.numel() != 0 {
            return Err(super::point_root_session::invalid(
                "selected-side anchored discharge and invariant product are disjoint. Discard the inconsistent point",
            ));
        }
        let mut discharge_physical = a.bounds(
            &c.where_cond(&captured_identity, anchored.lower(), intersection.lower())?,
            &c.where_cond(&captured_identity, anchored.upper(), intersection.upper())?,
        )?;
        // A star state takes the two-sided Q* enclosure, not the left value at the selected area.
        if let Some(star) = &star_discharge {
            discharge_physical = a.bounds(
                &c.where_cond(&captured_identity, anchored.lower(), star.lower())?,
                &c.where_cond(&captured_identity, anchored.upper(), star.upper())?,
            )?;
        }
        let projection = choose(
            a,
            &discharge_evaluation,
            &discharge_physical,
            Some(&captured_identity),
        )?;
        let discharge = projection.chosen;
        let pressure = selected_pressure(a, sources, &operand.parent, &query, side.index())?;
        let (velocity_arithmetic, momentum_arithmetic) =
            super::point_flux_arithmetic::momentum(a, &area, &discharge, &pressure.chosen)?;
        let velocity = center(a, &velocity_arithmetic)?;
        let momentum = center(a, &momentum_arithmetic)?;
        let selected_q = a.expansion_interval(&discharge)?;
        let discharge_discrepancy = a.sub(&discharge_physical, &selected_q)?;
        let eval_u = a.div_positive_interval(&selected_q, &area_bounds)?;
        let evaluation = a.mul(&selected_q, &eval_u)?;
        let retained_advection = a.expansion_mul(&discharge, &velocity_arithmetic)?;
        let evaluation =
            super::point_flux_arithmetic::intersect_advection(a, &evaluation, &retained_advection)?;
        let momentum_evaluation = a.add(&evaluation, &pressure.evaluation)?;
        let momentum_evaluation = super::point_flux_quantum::momentum(
            a,
            super::point_flux_quantum::MomentumInputs {
                area: &area_bounds,
                discharge: &selected_q,
                depth_upper: state.depth.upper(),
                pressure: &pressure.evaluation,
                gravity: sources.evaluation.config.gravity_m_per_s2,
            },
            &momentum_evaluation,
        )?;
        let arithmetic = a.expansion_interval(&momentum_arithmetic)?;
        let numerical = a.expansion_interval(&momentum)?;
        let momentum_projection = a.sub(&arithmetic, &numerical)?;
        Ok(SelectedPointFlux {
            candidate: operand.candidate,
            kind: operand.kind,
            invocation_owner: copy_tensor(a, &operand.owners)?,
            parent_rows: copy_tensor(a, &operand.parent)?,
            query,
            area,
            discharge,
            velocity,
            momentum,
            depth: state.depth,
            pressure: Some(pressure),
            area_evaluation,
            area_discrepancy,
            area_projection,
            discharge_evaluation,
            discharge_invariant_velocity,
            discharge_product,
            discharge_physical,
            discharge_proposal: projection.proposal,
            discharge_proposal_contained: projection.contained,
            discharge_selection_change: projection.change,
            discharge_discrepancy,
            velocity_arithmetic,
            momentum_arithmetic,
            momentum_evaluation,
            momentum_projection,
            reflected: Tensor::zeros(operand.owners.shape(), a.dtype(), c.device())?,
        })
    })
}

fn fan_discharge<R>(
    sources: &mut PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &crate::channel::section::SupportedAreaQuery<R>,
    side: SonicSide,
    numerical: &super::source_sonic_numerical::SourceSonicNumericalEvidence<R>,
) -> HydraulicResult<(
    solvr::common::arithmetic::CertifiedExpansion<R>,
    solvr::common::arithmetic::CertifiedExpansion<R>,
)>
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
    let c = sources.geometry.client();
    let stage = sources.select_rows(parent, query)?;
    let (velocity, change) =
        super::source_sonic_policy::invariant(sources, parent, query, side, numerical)?;
    checked::<R, _>(c, "selected fan anchored discharge", |a| {
        let source_velocity = if side == SonicSide::Left {
            &stage.velocity.left
        } else {
            &stage.velocity.right
        };
        let input = stage.velocity.input();
        let source_q = if side == SonicSide::Left {
            &input.left_discharge
        } else {
            &input.right_discharge
        };
        let delta = source_difference(
            a,
            &stage.bands,
            query,
            &stage.sources[side.index()],
            &stage.boundaries,
        )?
        .lower;
        let area = query.retained_endpoint_area(a, &stage.bands, &stage.boundaries)?;
        let area_term = a.expansion_mul(&delta, source_velocity)?;
        let change_term = a.expansion_mul(&area, &change)?;
        let change = a.expansion_add(&area_term, &change_term)?;
        Ok((a.expansion_add(source_q, &change)?, velocity))
    })
}

fn dry<R>(
    sources: &PairedSources<'_, R>,
    operand: Operands<R>,
) -> HydraulicResult<SelectedPointFlux<R>>
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
    checked::<R, _>(sources.geometry.client(), "exact dry point flux", |a| {
        let zero = Tensor::zeros(operand.owners.shape(), a.dtype(), a.client().device())?;
        macro_rules! expansion {
            () => {
                a.expansion_point(&zero)?
            };
        }
        macro_rules! interval {
            () => {
                a.point(&zero)?
            };
        }
        let momentum_evaluation = dry_momentum(a, &operand)?;
        Ok(SelectedPointFlux {
            candidate: operand.candidate,
            kind: PointFluxKind::Dry,
            invocation_owner: copy_tensor(a, &operand.owners)?,
            parent_rows: copy_tensor(a, &operand.parent)?,
            query: operand.query,
            area: expansion!(),
            discharge: expansion!(),
            velocity: expansion!(),
            momentum: expansion!(),
            depth: interval!(),
            pressure: None,
            area_evaluation: expansion!(),
            area_discrepancy: interval!(),
            area_projection: interval!(),
            discharge_evaluation: expansion!(),
            discharge_invariant_velocity: expansion!(),
            discharge_product: interval!(),
            discharge_physical: interval!(),
            discharge_proposal: expansion!(),
            discharge_proposal_contained: Tensor::ones(
                operand.owners.shape(),
                a.dtype(),
                a.client().device(),
            )?,
            discharge_selection_change: expansion!(),
            discharge_discrepancy: interval!(),
            velocity_arithmetic: expansion!(),
            momentum_arithmetic: expansion!(),
            momentum_evaluation,
            momentum_projection: interval!(),
            reflected: zero.clone(),
        })
    })
}

/// Momentum flux bounds of dry rows: `[0, upper]`, or exactly 0 without an upper bound.
pub(super) fn dry_momentum<R>(
    a: &mut Arithmetic<'_, R>,
    operand: &Operands<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let zero = Tensor::zeros(operand.owners.shape(), a.dtype(), a.client().device())?;
    Ok(match &operand.momentum_upper {
        Some(upper) => a.bounds(&zero, upper)?,
        None => a.point(&zero)?,
    })
}
