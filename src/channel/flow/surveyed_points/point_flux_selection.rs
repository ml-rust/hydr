use super::{
    paired_sources::PairedSources,
    point_flux_operands::{Operands, copy_tensor},
    point_flux_pressure::selected as selected_pressure,
    point_flux_projection::{center, choose},
    point_flux_types::{PointFluxKind, SelectedPointFlux},
    potential_range::{PotentialContext, PotentialEvaluation},
    source_sonic_types::SonicSide,
};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
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
use solvr::integrate::CertifiedSimpsonAlgorithms;

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
            let parameter = a.point(&midpoint)?;
            let endpoint = operand
                .query
                .with_parameter_bounds(a, &stage.bands, &parameter)?;
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
    let (discharge_evaluation, discharge_invariant_velocity, captured_identity) =
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
                ))
            })?
        } else if let PointFluxKind::Fan(side) = operand.kind {
            let (discharge, velocity) = fan_discharge(sources, &operand.parent, &query, side)?;
            let identity =
                Tensor::zeros(operand.parent.shape(), discharge.high().dtype(), c.device())?;
            (discharge, velocity, identity)
        } else {
            let endpoint = super::wet_star_endpoint::evaluate(
                sources,
                &operand.parent,
                &query,
                PhysicalEndpoint::Lower,
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
                Ok((
                    discharge,
                    a.expansion_bounds(velocity.high(), velocity.low(), velocity.abs_error())?,
                    copy_tensor(a, &stage.velocity.uniform)?,
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
        let discharge_physical = a.bounds(
            &c.where_cond(&captured_identity, anchored.lower(), intersection.lower())?,
            &c.where_cond(&captured_identity, anchored.upper(), intersection.upper())?,
        )?;
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
        })
    })
}

fn fan_discharge<R>(
    sources: &mut PairedSources<'_, R>,
    parent: &Tensor<R>,
    query: &crate::channel::section::SupportedAreaQuery<R>,
    side: SonicSide,
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
    let empty = Tensor::empty(&[0], DType::I64, c.device())?;
    let [left, right] = if side == SonicSide::Left {
        sources.source_values([parent, &empty])?
    } else {
        sources.source_values([&empty, parent])?
    };
    let potential = if side == SonicSide::Left { left } else { right };
    let stage = sources.select_rows(parent, query)?;
    let epsilon = checked::<R, _>(c, "selected fan source invariant", |a| {
        if side == SonicSide::Left {
            Ok(a.expansion_add(&stage.velocity.left, &potential.retained)?)
        } else {
            Ok(a.expansion_sub(&potential.retained, &stage.velocity.right)?)
        }
    })?;
    let context = PotentialContext::new(
        sources.geometry,
        &stage.bands,
        &stage.sources[side.index()],
        query,
        &stage.boundaries,
        PotentialEvaluation {
            config: sources.evaluation.config,
            panels: sources.evaluation.panels,
            work: sources.evaluation.work,
        },
    )?;
    let (velocity, change) = super::source_sonic_invariant::endpoint(&context, side, &epsilon)?;
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
            momentum_evaluation: interval!(),
            momentum_projection: interval!(),
        })
    })
}
