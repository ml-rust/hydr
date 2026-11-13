use super::{
    paired_sources::PairedSources,
    point_root_session::PointRootSession,
    potential_absolute::endpoint_potential,
    potential_increment::endpoint_increment,
    potential_range::{PotentialContext, PotentialEvaluation},
    source_sonic_types::{
        SonicFanMembership, SonicSide, SourceSonicCertificate, SourceSonicCertificates,
        SourceSonicDomain, SourceSonicEvidence, SourceSonicStatus,
    },
    wet_star::intersection as intersect,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::{gravity_depth_root, scalar},
        error::{Arithmetic, HydraulicResult, checked},
    },
    section::{PhysicalEndpoint, source_difference},
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

pub(super) fn evaluate<R>(
    session: &mut PointRootSession<'_, '_, R>,
) -> HydraulicResult<SourceSonicCertificates<R>>
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
    let mut domains = Vec::new();
    for request in &session.sonic.requests {
        if request.terminal {
            continue;
        }
        for domain in &request.domains {
            domains.push(certificate(session.sources, &Operands::from(domain))?);
        }
    }
    let c = session.sources.geometry.client();
    let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let result = Tensor::empty(value.shape(), value.dtype(), c.device())?;
        c.copy_into(&result, value)?;
        Ok(result)
    };
    let evidence = checked::<R, _>(c, "source sonic owned endpoint evidence", |a| {
        let mut evidence = Vec::with_capacity(session.sonic.requests.len());
        for request in &session.sonic.requests {
            evidence.push(SourceSonicEvidence {
                invocation_owner: copy(&request.binding.owners)?,
                parent_rows: copy(&request.binding.rows)?,
                side: request.binding.side,
                status: copy(&request.status)?,
                dry_residual: owned_curve(a, &request.dry_residual)?,
                source_residual: owned_curve(a, &request.source_residual)?,
                endpoint_points: request.endpoint_points,
                bracket_points: request.bracket_points,
                bracket_expansions: request.bracket_expansions,
            });
        }
        Ok(evidence)
    })?;
    Ok(SourceSonicCertificates {
        invocation_rows: copy(&session.invocation_rows)?,
        status: [
            copy(&session.sonic.status[0])?,
            copy(&session.sonic.status[1])?,
        ],
        domains,
        evidence,
        callback_points: session.callback_points(),
        point_limit: session.point_limit(),
    })
}

struct Operands<'a, R: Runtime<DType = DType>> {
    invocation_owner: &'a Tensor<R>,
    parent_rows: &'a Tensor<R>,
    side: SonicSide,
    query: &'a crate::channel::section::SupportedAreaQuery<R>,
    epsilon: &'a solvr::common::arithmetic::CertifiedExpansion<R>,
    classification: SourceSonicStatus,
    depth: &'a solvr::common::arithmetic::Interval<R>,
    normalized_depth: &'a solvr::common::arithmetic::Interval<R>,
    status: &'a Tensor<R>,
    iterations: usize,
    callback_points: usize,
    mapped_points: usize,
}
impl<'a, R: Runtime<DType = DType>> From<&'a SourceSonicDomain<R>> for Operands<'a, R> {
    fn from(d: &'a SourceSonicDomain<R>) -> Self {
        Self {
            invocation_owner: &d.invocation_owner,
            parent_rows: &d.parent_rows,
            side: d.side,
            query: &d.query,
            epsilon: &d.epsilon,
            classification: d.classification,
            depth: &d.depth,
            normalized_depth: &d.normalized_depth,
            status: &d.status,
            iterations: d.iterations,
            callback_points: d.callback_points,
            mapped_points: d.mapped_points,
        }
    }
}

/// Select immutable certificate operands without selecting or copying native continuation state.
pub(super) fn selected<R>(
    session: &mut PointRootSession<'_, '_, R>,
    demanded: [&Tensor<R>; 2],
) -> HydraulicResult<Vec<SourceSonicCertificate<R>>>
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
    session
        .sources
        .check_root_witness(session.invocation.source_witness())?;
    session.check_domains()?;
    if session
        .sonic
        .requests
        .iter()
        .any(|request| request.terminal)
    {
        return Err(super::point_root_session::invalid(
            "selected sonic certificates remain sealed after an earlier numerical error",
        ));
    }
    let c = session.sources.geometry.client();
    let mut output = Vec::new();
    for request in &session.sonic.requests {
        for domain in &request.domains {
            if domain.classification != SourceSonicStatus::PositiveRoot {
                continue;
            }
            let mask =
                c.index_select(demanded[domain.side.index()], 0, &domain.invocation_owner)?;
            let rows = c.nonzero(&mask)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() == 0 {
                continue;
            }
            let (owners, parent, query, epsilon, depth, normalized, status) =
                checked::<R, _>(c, "source sonic demanded certificate operands", |a| {
                    Ok((
                        c.index_select(&domain.invocation_owner, 0, &rows)?,
                        c.index_select(&domain.parent_rows, 0, &rows)?,
                        domain.query.select(a, &rows)?,
                        super::potential_range::select_expansion(a, &domain.epsilon, &rows)?,
                        crate::channel::flow::surveyed::equation_ops::select_interval(
                            a,
                            &domain.depth,
                            &rows,
                        )?,
                        crate::channel::flow::surveyed::equation_ops::select_interval(
                            a,
                            &domain.normalized_depth,
                            &rows,
                        )?,
                        c.index_select(&domain.status, 0, &rows)?,
                    ))
                })?;
            output.push(certificate(
                session.sources,
                &Operands {
                    invocation_owner: &owners,
                    parent_rows: &parent,
                    side: domain.side,
                    query: &query,
                    epsilon: &epsilon,
                    classification: domain.classification,
                    depth: &depth,
                    normalized_depth: &normalized,
                    status: &status,
                    iterations: domain.iterations,
                    callback_points: domain.callback_points,
                    mapped_points: domain.mapped_points,
                },
            )?);
        }
    }
    Ok(output)
}

fn certificate<R>(
    sources: &mut PairedSources<'_, R>,
    domain: &Operands<'_, R>,
) -> HydraulicResult<SourceSonicCertificate<R>>
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
    let selected = sources.select_rows(domain.parent_rows, domain.query)?;
    let mut endpoint_velocity = Vec::with_capacity(2);
    let mut endpoint_change = Vec::with_capacity(2);
    let mut endpoint_residual = Vec::with_capacity(2);
    for endpoint in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
        let query = checked::<R, _>(c, "source sonic certificate endpoint query", |a| {
            domain.query.endpoint(a, &selected.bands, endpoint)
        })?;
        let context = PotentialContext::new(
            sources.geometry,
            &selected.bands,
            &selected.sources[domain.side.index()],
            &query,
            &selected.boundaries,
            PotentialEvaluation {
                config: sources.evaluation.config,
                panels: sources.evaluation.panels,
                work: sources.evaluation.work,
            },
        )?;
        let potential = endpoint_potential(&context, PhysicalEndpoint::Lower)?;
        let increment = endpoint_increment(&context, PhysicalEndpoint::Lower)?;
        let (velocity, change) =
            checked::<R, _>(c, "source sonic retained trace invariant", |a| {
                if domain.side == SonicSide::Left {
                    let zero = a.expansion_point(&Tensor::zeros(
                        domain.parent_rows.shape(),
                        a.dtype(),
                        c.device(),
                    )?)?;
                    Ok((
                        a.expansion_sub(domain.epsilon, &potential.retained)?,
                        a.expansion_sub(&zero, &increment.retained)?,
                    ))
                } else {
                    Ok((
                        a.expansion_sub(&potential.retained, domain.epsilon)?,
                        increment.retained,
                    ))
                }
            })?;
        endpoint_velocity.push(velocity);
        endpoint_change.push(change);
        endpoint_residual.push(super::source_sonic_residual::endpoint(
            sources,
            domain.parent_rows,
            &query,
            PhysicalEndpoint::Lower,
            domain.side,
            domain.epsilon,
        )?);
    }
    checked::<R, _>(c, "source sonic physical certificates", |a| {
        let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
            c.copy_into(&owned, value)?;
            Ok(owned)
        };
        let parameter = a.bounds(
            &copy(domain.query.query.parameter.lower())?,
            &copy(domain.query.query.parameter.upper())?,
        )?;
        let query = domain
            .query
            .with_parameter_bounds(a, &selected.bands, &parameter)?;
        let zero = Tensor::zeros(domain.parent_rows.shape(), a.dtype(), c.device())?;
        let sound = if domain.classification == SourceSonicStatus::DryContact {
            a.point(&zero)?
        } else {
            let state = sources
                .geometry
                .recover_query(a, &selected.bands, &query)?
                .state;
            let gravity = scalar(a, sources.evaluation.config.gravity_m_per_s2)?;
            gravity_depth_root(a, &gravity, &state.hydraulic_depth, None)?
        };
        let sonic_velocity = if domain.side == SonicSide::Left {
            a.bounds(sound.lower(), sound.upper())?
        } else {
            let z = a.point(&zero)?;
            a.sub(&z, &sound)?
        };
        let first = a.expansion_interval(&endpoint_velocity[0])?;
        let last = a.expansion_interval(&endpoint_velocity[1])?;
        let invariant = if domain.side == SonicSide::Left {
            a.bounds(last.lower(), first.upper())?
        } else {
            a.bounds(first.lower(), last.upper())?
        };
        // Sonic and invariant velocities refer to the same established source-curve root.
        let (velocity, velocity_valid) = intersect(a, &sonic_velocity, &invariant)?;
        let input = selected.velocity.input();
        let source_velocity = if domain.side == SonicSide::Left {
            &selected.velocity.left
        } else {
            &selected.velocity.right
        };
        let source_discharge = if domain.side == SonicSide::Left {
            &input.left_discharge
        } else {
            &input.right_discharge
        };
        let difference = source_difference(
            a,
            &selected.bands,
            &query,
            &selected.sources[domain.side.index()],
            &selected.boundaries,
        )?;
        let lower = a.expansion_interval(&difference.lower)?;
        let upper = a.expansion_interval(&difference.upper)?;
        let difference = a.bounds(lower.lower(), upper.upper())?;
        let area_difference = a.expansion_from_interval(&difference)?;
        let first = a.expansion_interval(&endpoint_change[0])?;
        let last = a.expansion_interval(&endpoint_change[1])?;
        let change = a.bounds(
            &c.minimum(first.lower(), last.lower())?,
            &c.maximum(first.upper(), last.upper())?,
        )?;
        let velocity_change = a.expansion_from_interval(&change)?;
        let area = a.expansion_from_interval(&query.area)?;
        let area_term = a.expansion_mul(&area_difference, source_velocity)?;
        let velocity_term = a.expansion_mul(&area, &velocity_change)?;
        let discharge_change = a.expansion_add(&area_term, &velocity_term)?;
        let anchored_discharge = a.expansion_add(source_discharge, &discharge_change)?;
        let anchored = a.expansion_interval(&anchored_discharge)?;
        let product = a.mul(&query.area, &velocity)?;
        let (mut discharge, discharge_valid) = intersect(a, &anchored, &product)?;
        let mut valid = c.mul(&velocity_valid, &discharge_valid)?;
        if domain.classification == SourceSonicStatus::SourceContact {
            let captured = a.expansion_interval(source_discharge)?;
            let (exact, exact_valid) = intersect(a, &discharge, &captured)?;
            discharge = exact;
            valid = c.mul(&valid, &exact_valid)?;
        }
        let residual_lower = endpoint_residual[0].physical.lower();
        let residual_upper = endpoint_residual[1].physical.upper();
        let residual_ordered = c.le(residual_lower, residual_upper)?;
        let residual = a.bounds(
            &c.where_cond(&residual_ordered, residual_lower, &zero)?,
            &c.where_cond(&residual_ordered, residual_upper, &zero)?,
        )?;
        valid = c.mul(
            &valid,
            &c.mul(
                &residual_ordered,
                &c.mul(
                    &c.le(residual.lower(), &zero)?,
                    &c.ge(residual.upper(), &zero)?,
                )?,
            )?,
        )?;
        valid = c.mul(
            &valid,
            &c.mul(
                &endpoint_residual[0].available,
                &endpoint_residual[1].available,
            )?,
        )?;
        let status = c.where_cond(
            &valid,
            &Tensor::zeros(domain.parent_rows.shape(), DType::I64, c.device())?,
            &Tensor::full_scalar(domain.parent_rows.shape(), DType::I64, 2.0, c.device())?,
        )?;
        let velocity = a.bounds(
            &c.where_cond(&valid, velocity.lower(), &zero)?,
            &c.where_cond(&valid, velocity.upper(), &zero)?,
        )?;
        let discharge = a.bounds(
            &c.where_cond(&valid, discharge.lower(), &zero)?,
            &c.where_cond(&valid, discharge.upper(), &zero)?,
        )?;
        Ok(SourceSonicCertificate {
            invocation_owner: copy(domain.invocation_owner)?,
            parent_rows: copy(domain.parent_rows)?,
            side: domain.side,
            query,
            depth: a.bounds(&copy(domain.depth.lower())?, &copy(domain.depth.upper())?)?,
            normalized_depth: a.bounds(
                &copy(domain.normalized_depth.lower())?,
                &copy(domain.normalized_depth.upper())?,
            )?,
            sound,
            velocity,
            discharge,
            source_discharge: a.expansion_bounds(
                source_discharge.high(),
                source_discharge.low(),
                source_discharge.abs_error(),
            )?,
            area_difference,
            velocity_change,
            discharge_change,
            anchored_discharge,
            residual,
            classification: domain.classification,
            status,
            depth_status: copy(domain.status)?,
            iterations: domain.iterations,
            callback_points: domain.callback_points,
            mapped_points: domain.mapped_points,
            fan_membership: SonicFanMembership::NotChecked,
        })
    })
}

fn owned_curve<R>(
    a: &mut Arithmetic<'_, R>,
    value: &super::wave_curve_types::CurveValue<R>,
) -> HydraulicResult<super::wave_curve_types::CurveValue<R>>
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
    let copy = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        let owned = Tensor::empty(value.shape(), value.dtype(), c.device())?;
        c.copy_into(&owned, value)?;
        Ok(owned)
    };
    Ok(super::wave_curve_types::CurveValue {
        retained: a.expansion_bounds(
            &copy(value.retained.high())?,
            &copy(value.retained.low())?,
            &copy(value.retained.abs_error())?,
        )?,
        physical: a.bounds(
            &copy(value.physical.lower())?,
            &copy(value.physical.upper())?,
        )?,
        available: copy(&value.available)?,
    })
}
