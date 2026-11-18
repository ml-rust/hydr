use super::potential_absolute::admitted_analytic;
use super::potential_absolute::prefix_potential;
use super::potential_analytic;
use super::potential_numerical::wet_contributions;
use super::potential_range::{
    PotentialContext, PotentialValue, scatter, select_expansion, zero_value,
};
use super::source_terms::SourceRows;
use crate::channel::{
    flow::surveyed::{
        equation_ops::select_interval,
        error::{HydraulicResult, checked},
    },
    section::{ConditionalBandSegments, PhysicalEndpoint, SupportedAreaQuery, source_difference},
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

/// Integrate source-to-endpoint segments directly. Absolute leaves remain unevaluated.
pub(super) fn endpoint_increment<R>(
    context: &PotentialContext<'_, R>,
    endpoint: PhysicalEndpoint,
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
    // Kept source terms leave the fresh source query to the reversed dry prefix alone.
    let source = match &context.source_rows {
        Some(rows) => SourceSide::Kept(rows),
        None => SourceSide::Fresh(Box::new(context.source_query()?)),
    };
    let (target_query, target_area, delta, equal, nonzero_rows) =
        checked(c, "point signed potential dispatch", |a| {
            let target_query = context.query.endpoint(a, context.bands, endpoint)?;
            let target_area =
                target_query.retained_endpoint_area(a, context.bands, context.boundaries)?;
            let delta = source_difference(
                a,
                context.bands,
                &target_query,
                context.source,
                context.boundaries,
            )?
            .lower;
            let n = context.bands.section_owner.numel();
            let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
            let equal = exact_zero(c, &delta, &zero)?;
            let nonzero_rows = c.nonzero(&c.eq(&equal, &zero)?)?.flatten()?;
            Ok((target_query, target_area, delta, equal, nonzero_rows))
        })?;
    if let SourceSide::Kept(rows) = &source
        && nonzero_rows.numel() != 0
    {
        rows.ensure(context, &nonzero_rows)?;
    }
    let (mut result, dry_forward, dry_reverse, wet) =
        checked(c, "point signed potential dispatch", |a| {
            let n = context.bands.section_owner.numel();
            let zero = Tensor::zeros(&[n], DType::F64, c.device())?;
            let mut result = zero_value(a, n)?;
            let mut analytic_mask = zero.clone();
            if nonzero_rows.numel() != 0 {
                let bands = context.bands.select(a, &nonzero_rows)?;
                let target_query = target_query.select(a, &nonzero_rows)?;
                let gravity = context.evaluation.config.gravity_m_per_s2;
                let (source_width, source) = match &source {
                    SourceSide::Kept(rows) => {
                        let terms = rows.terms(a, &nonzero_rows)?;
                        (terms.top_width, terms.analytic)
                    }
                    SourceSide::Fresh(source_query) => {
                        let source_query = source_query.select(a, &nonzero_rows)?;
                        let source_area = select_expansion(
                            a,
                            context.source.requested_areas_m2(),
                            &nonzero_rows,
                        )?;
                        let source_state = context
                            .geometry
                            .recover_query(a, &bands, &source_query)?
                            .state;
                        let source = admitted_analytic(
                            a,
                            &bands,
                            &source_area,
                            &source_state.depth,
                            gravity,
                        )?;
                        (source_state.top_width, source)
                    }
                };
                let target_area = select_expansion(a, &target_area, &nonzero_rows)?;
                let delta = select_expansion(a, &delta, &nonzero_rows)?;
                let target_state = context
                    .geometry
                    .recover_query(a, &bands, &target_query)?
                    .state;
                let target =
                    admitted_analytic(a, &bands, &target_area, &target_state.depth, gravity)?;
                let analytic = potential_analytic::difference(
                    a,
                    &source,
                    &target,
                    &source_width,
                    &target_state.top_width,
                    &delta,
                    gravity,
                )?;
                let local_zero = Tensor::zeros(&[nonzero_rows.numel()], DType::F64, c.device())?;
                let local = c
                    .nonzero(&c.ne(&analytic.eligible, &local_zero)?)?
                    .flatten()?;
                let analytic_rows = c.index_select(&nonzero_rows, 0, &local)?;
                let value = PotentialValue {
                    retained: select_expansion(a, &analytic.retained_increment, &local)?,
                    physical: select_interval(a, &analytic.physical_increment, &local)?,
                    analytic: None,
                };
                result = scatter(a, result, &analytic_rows, &value)?;
                analytic_mask = c.index_put(&zero, 0, &nonzero_rows, &analytic.eligible)?;
            }
            let remaining = c.eq(&c.maximum(&equal, &analytic_mask)?, &zero)?;
            let source_dry = exact_zero(c, context.source.requested_areas_m2(), &zero)?;
            let target_dry = exact_zero(c, &target_area, &zero)?;
            let forward = c.nonzero(&c.mul(&remaining, &source_dry)?)?.flatten()?;
            let reverse = c.nonzero(&c.mul(&remaining, &target_dry)?)?.flatten()?;
            let wet = c
                .nonzero(&c.mul(
                    &remaining,
                    &c.eq(&c.maximum(&source_dry, &target_dry)?, &zero)?,
                )?)?
                .flatten()?;
            Ok((result, forward, reverse, wet))
        })?;
    let source_query = match source {
        SourceSide::Fresh(query) => Some(*query),
        SourceSide::Kept(_) if dry_reverse.numel() != 0 => Some(context.source_query()?),
        SourceSide::Kept(_) => None,
    };
    for (rows, query, reverse) in [
        (&dry_forward, Some(&target_query), false),
        (&dry_reverse, source_query.as_ref(), true),
    ] {
        let Some(query) = query.filter(|_| rows.numel() != 0) else {
            continue;
        };
        let (bands, query, boundaries) = checked(c, "point signed dry prefix selection", |a| {
            Ok((
                context.bands.select(a, rows)?,
                query.select(a, rows)?,
                context.boundaries.select(a, rows)?,
            ))
        })?;
        let mut value = prefix_potential(context, &bands, &query, &boundaries)?;
        if reverse {
            value = checked(c, "point reversed dry prefix", |a| {
                let zero = zero_value(a, rows.numel())?;
                Ok(PotentialValue {
                    retained: a.expansion_sub(&zero.retained, &value.retained)?,
                    physical: a.sub(&zero.physical, &value.physical)?,
                    analytic: None,
                })
            })?;
        }
        result = checked(c, "point signed dry prefix assembly", |a| {
            scatter(a, result, rows, &value)
        })?;
    }
    if wet.numel() != 0 {
        let segments = checked(c, "point signed potential segments", |a| {
            ConditionalBandSegments::prepare(
                a,
                context.bands,
                context.source,
                &target_query,
                context.boundaries,
            )
        })?;
        let value = wet_contributions(context, &segments, &wet)?;
        result = checked(c, "point signed potential assembly", |a| {
            scatter(a, result, &wet, &value)
        })?;
    }
    Ok(result)
}

/// The source of the source terms of the nonzero rows.
enum SourceSide<'q, 'a, R: Runtime<DType = DType>> {
    /// Kept per-row terms of a fixed paired side.
    Kept(&'q SourceRows<'a, R>),
    /// The source query of every context row.
    Fresh(Box<SupportedAreaQuery<R>>),
}

/// One where every component of `value` is exactly zero.
fn exact_zero<R>(
    c: &R::Client,
    value: &CertifiedExpansion<R>,
    zero: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: CompareOps<R> + BinaryOps<R>,
{
    Ok(c.mul(
        &c.eq(value.high(), zero)?,
        &c.mul(&c.eq(value.low(), zero)?, &c.eq(value.abs_error(), zero)?)?,
    )?)
}

#[cfg(test)]
pub(super) mod film {
    use super::super::potential_absolute::{endpoint_potential, source_potential};
    use super::super::potential_increment_fixtures as frozen;
    use super::super::potential_range::PotentialEvaluation;
    use super::super::potential_range_fixture::{check, check_value, config, section};
    use super::super::source_fixture_support::{
        self as support, Client, Result, components, endpoints, expansion, hydraulic,
    };
    use super::*;
    use crate::channel::{
        flow::surveyed::{error, error::HydraulicError, work::WorkLedger},
        section::{
            AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry, PrismaticRecoveryConfig,
        },
    };

    fn film_target<R>(
        client: &R::Client,
        context: &PotentialContext<'_, R>,
        endpoint: PhysicalEndpoint,
        reference: &frozen::original_film::Target,
    ) -> Result<PotentialValue<R>>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let value = hydraulic(endpoint_potential(context, endpoint))?;
        check_value::<R>(client, &value, reference.potential, reference.id)?;
        let increment = hydraulic(endpoint_increment(context, endpoint))?;
        check_value::<R>(client, &increment, reference.left_increment, reference.id)?;
        check_value::<R>(client, &increment, reference.right_increment, reference.id)?;
        let target = value
            .analytic
            .as_ref()
            .ok_or("Missing film analytic target")?;
        for actual in endpoints(&support::bounds::<R>(client, &target.sound)?)? {
            check(actual, reference.sound, reference.id);
        }
        Ok(value)
    }

    pub(in crate::channel::flow::surveyed_points) fn run<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let inputs: &frozen::original_film::Sources = &frozen::original_film::SOURCES;
        let bracket: &frozen::original_film::SonicBracket = &frozen::original_film::SONIC_BRACKET;
        let section = section::<R>(client, &[frozen::original_film::PROFILE])?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let owners = Tensor::from_slice(&[0_i64; 2], &[2], client.device())?;
        let area = expansion::<R>(client, &inputs.areas.map(|a| [a[0], a[1], 0]), true)?;
        let discharge = expansion::<R>(client, &inputs.discharges.map(|q| [q[0], q[1], 0]), true)?;
        let source = section.recover_area_points_tensor(
            &owners,
            &area,
            &PrismaticRecoveryConfig::default(),
        )?;
        let config = config(inputs.gravity_bits);
        let work = WorkLedger::new(&config);
        let (bands, boundaries, epsilon) = hydraulic(error::checked::<R, _>(
            client,
            "original film invariant",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let velocity = a.expansion_div_positive(&discharge, &area)?;
                let velocity_points =
                    components(&velocity).map_err(|e| HydraulicError::Numerical {
                        context: e.to_string(),
                    })?;
                for (actual, expected) in velocity_points.into_iter().zip(inputs.velocities) {
                    assert_eq!(actual, [expected[0], expected[1], 0]);
                }
                let left = Tensor::from_slice(&[0_i64; 2], &[2], client.device())?;
                let right = Tensor::from_slice(&[1_i64; 2], &[2], client.device())?;
                let right_velocity =
                    super::super::potential_range::select_expansion(a, &velocity, &right)?;
                let left_velocity =
                    super::super::potential_range::select_expansion(a, &velocity, &left)?;
                let jump = a.expansion_sub(&right_velocity, &left_velocity)?;
                assert_eq!(
                    components(&jump).map_err(|e| HydraulicError::Numerical {
                        context: e.to_string()
                    })?,
                    [[inputs.velocity_jump[0], inputs.velocity_jump[1], 0]; 2]
                );
                let four =
                    a.expansion_point(&Tensor::from_slice(&[4.0; 2], &[2], client.device())?)?;
                let epsilon = a.expansion_sub(&four, &jump)?;
                Ok((bands, boundaries, epsilon))
            },
        ))?;
        assert_eq!(
            components(&epsilon)?,
            [[inputs.epsilon[0], inputs.epsilon[1], 0]; 2]
        );
        for star in [true, false] {
            let query = hydraulic(error::checked::<R, _>(
                client,
                "original film target chart",
                |a| {
                    let (scale, lo, hi) = if star {
                        let point = frozen::original_film::STAR.declared_point_input.ok_or_else(|| HydraulicError::Numerical {
                    context: "film star lacks its frozen represented point. Check reference declaration".into() })?;
                        (point, 1.0_f64.to_bits(), 1.0_f64.to_bits())
                    } else {
                        (
                            bracket.scale_components,
                            bracket.parameter_bounds[0],
                            bracket.parameter_bounds[1],
                        )
                    };
                    let scale = a.expansion_bounds(
                        &Tensor::from_slice(&[f64::from_bits(scale[0]); 2], &[2], client.device())?,
                        &Tensor::from_slice(&[f64::from_bits(scale[1]); 2], &[2], client.device())?,
                        &Tensor::zeros(&[2], DType::F64, client.device())?,
                    )?;
                    let parameter = a.bounds(
                        &Tensor::from_slice(&[f64::from_bits(lo); 2], &[2], client.device())?,
                        &Tensor::from_slice(&[f64::from_bits(hi); 2], &[2], client.device())?,
                    )?;
                    AreaQuery::band_affine(
                        a,
                        &bands,
                        AreaAnchor::Zero,
                        AreaDirection::Add,
                        &scale,
                        &parameter,
                    )?
                    .prove_support(a, &bands)
                },
            ))?;
            let context = hydraulic(PotentialContext::new(
                &geometry,
                &bands,
                &source,
                &query,
                &boundaries,
                PotentialEvaluation {
                    config: &config,
                    panels: 4,
                    work: &work,
                },
            ))?;
            let value = hydraulic(source_potential(&context))?;
            for side in 0..2 {
                check(
                    endpoints(&value.physical)?[side],
                    inputs.potential[side],
                    inputs.case_id,
                );
                let analytic = value.analytic.as_ref().ok_or("Missing exact film source")?;
                check(
                    endpoints(&support::bounds::<R>(client, &analytic.sound)?)?[side],
                    inputs.sound[side],
                    inputs.case_id,
                );
            }
            if inputs.potential_identity_requires_zero_radius {
                assert_eq!(
                    components(&value.retained)?,
                    [[inputs.exact_potential_identity_bits, 0, 0]; 2]
                );
            }
            if star {
                let reference = &frozen::original_film::STAR;
                let value = film_target::<R>(client, &context, PhysicalEndpoint::Lower, reference)?;
                let residual =
                    hydraulic(error::checked::<R, _>(client, "film star residual", |a| {
                        let two = a.expansion_point(&Tensor::from_slice(
                            &[2.0; 2],
                            &[2],
                            client.device(),
                        )?)?;
                        let total = a.expansion_mul(&value.retained, &two)?;
                        let residual = a.expansion_sub(&total, &epsilon)?;
                        Ok(a.expansion_interval(&residual)?)
                    }))?;
                assert!(
                    reference.physical_residual_zero,
                    "{}",
                    reference.area_rational
                );
                for actual in endpoints(&residual)? {
                    check(actual, reference.residual, reference.id);
                }
            } else {
                assert!(
                    bracket.zero_anchor && bracket.add && bracket.strict_opposite_residual_signs
                );
                assert_eq!(bracket.exact_physical_parameter_rational, "1/9");
                let mut physical = Vec::new();
                for (endpoint, reference) in [
                    (PhysicalEndpoint::Lower, &bracket.endpoints[0]),
                    (PhysicalEndpoint::Upper, &bracket.endpoints[1]),
                ] {
                    let value = film_target::<R>(client, &context, endpoint, reference)?;
                    let residual = hydraulic(error::checked::<R, _>(
                        client,
                        "film directed sonic residual",
                        |a| {
                            super::super::potential_sonic::residual(
                                a,
                                value.analytic.as_ref().ok_or_else(|| {
                                    HydraulicError::Numerical {
                                        context:
                                            "film endpoint lacks analytic metadata. Check dispatch"
                                                .into(),
                                    }
                                })?,
                                &epsilon,
                            )
                        },
                    ))?;
                    for actual in endpoints(&residual.physical_residual)? {
                        check(actual, reference.residual, reference.id);
                        if matches!(endpoint, PhysicalEndpoint::Lower) {
                            assert!(actual[1] < 0.0, "{actual:?}");
                        } else {
                            assert!(actual[0] > 0.0, "{actual:?}");
                        }
                    }
                    physical.push(endpoints(&value.physical)?[0]);
                    assert!(
                        reference.declared_point_input.is_none(),
                        "{}",
                        reference.area_rational
                    );
                    let area =
                        hydraulic(error::checked::<R, _>(client, "film directed area", |a| {
                            let endpoint = query.endpoint(a, &bands, endpoint)?;
                            Ok(a.bounds(endpoint.area.lower(), endpoint.area.upper())?)
                        }))?;
                    let expected = reference.area_bounds.map(f64::from_bits);
                    let actual = endpoints(&area)?[0];
                    assert!(actual[0] <= expected[0] && expected[1] <= actual[1]);
                }
                let exact = &frozen::original_film::SONIC_PHYSICAL;
                let expected = exact.potential.bounds.map(f64::from_bits);
                assert!(
                    physical[0][0] <= expected[0] && expected[1] <= physical[1][1],
                    "{}",
                    exact.area_rational
                );
                assert!(exact.physical_residual_zero && exact.declared_point_input.is_none());
            }
        }
        assert_eq!(work.snapshot(), Default::default());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger};
    use crate::channel::flow::surveyed_points::potential_absolute::endpoint_potential;
    use crate::channel::flow::surveyed_points::potential_range::PotentialEvaluation;
    use crate::channel::flow::surveyed_points::source_fixture_support::{
        self as support, Client, Result, endpoints, hydraulic,
    };
    use crate::channel::section::{AreaAnchor, AreaDirection, AreaQuery, HydraulicGeometry};

    fn composed_underflow<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let section = support::batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        // The existing canonical rectangle has unit width and a zero bed elevation.
        let owners = Tensor::from_slice(&[1_i64], &[1], client.device())?;
        let source_area = support::expansion::<R>(client, &[[0; 3]], true)?;
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: 1.0,
            ..PrismaticRiemannConfig::default()
        };
        let source = section.recover_area_points_tensor(&owners, &source_area, &config.recovery)?;
        let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
            client,
            "composed positive underflow potential map",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let scale = a.expansion_point(&Tensor::from_slice(
                    &[f64::from_bits(1)],
                    &[1],
                    client.device(),
                )?)?;
                let parameter = a.point(&Tensor::from_slice(&[0.5], &[1], client.device())?)?;
                let query = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                for endpoint in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
                    let target = query.endpoint(a, &bands, endpoint)?;
                    let raw = target.retained_endpoint_area(a, &bands, &boundaries)?;
                    let high = raw.high().item::<f64>()?;
                    let low = raw.low().item::<f64>()?;
                    let radius = raw.abs_error().item::<f64>()?;
                    assert_eq!(high, 0.0);
                    assert_eq!(low, 0.0);
                    assert!(radius > 0.0);
                    let exact_dry = high == 0.0 && low == 0.0 && radius == 0.0;
                    assert!(!exact_dry);
                    let raw_bounds = a.expansion_interval(&raw)?;
                    assert!(raw_bounds.lower().item::<f64>()? < 0.0);
                    let physical = geometry.recover_query(a, &bands, &target)?.state;
                    assert_eq!(physical.area.lower().item::<f64>()?, 0.0);
                    assert!(physical.area.upper().item::<f64>()? > 0.0);
                    assert_eq!(physical.wet_possible.item::<f64>()?, 1.0);
                }
                Ok((bands, boundaries, query))
            },
        ))?;
        let work = WorkLedger::new(&config);
        let context = hydraulic(PotentialContext::new(
            &geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config: &config,
                panels: 4,
                work: &work,
            },
        ))?;
        // A=2^-1075 gives Phi=2*sqrt(A)=sqrt(2)*2^-537.
        // Thus 2^-537 < Phi < 2^-536, using exact normal powers of two.
        let truth_lower = f64::from_bits((1023_u64 - 537) << 52);
        let truth_upper = f64::from_bits((1023_u64 - 536) << 52);
        for endpoint in [PhysicalEndpoint::Lower, PhysicalEndpoint::Upper] {
            let absolute =
                hydraulic(endpoint_potential(&context, endpoint).inspect_err(|error| {
                    eprintln!("composed underflow endpoint_potential: {error:?}");
                }))?;
            let increment =
                hydraulic(endpoint_increment(&context, endpoint).inspect_err(|error| {
                    eprintln!("composed underflow endpoint_increment: {error:?}");
                }))?;
            for (name, value) in [("absolute", &absolute), ("increment", &increment)] {
                if let Some(analytic) = &value.analytic {
                    assert_eq!(analytic.eligible.item::<f64>()?, 0.0, "{name}");
                }
                let projected = support::bounds::<R>(client, &value.retained)?;
                for interval in [&value.physical, &projected] {
                    let actual = endpoints(interval)?[0];
                    assert!(
                        actual[0].is_finite() && actual[1].is_finite(),
                        "{name}: {actual:?}"
                    );
                    assert!(
                        actual[0] <= truth_lower && truth_upper <= actual[1],
                        "{name}: {actual:?} excludes the independent power-of-two enclosure"
                    );
                    assert!(actual[1] > 0.0, "{name}: {actual:?}");
                }
                assert!(value.physical.lower().item::<f64>()? >= 0.0, "{name}");
            }
            hydraulic(error::checked::<R, _>(
                client,
                "retained underflow input after composition",
                |a| {
                    let target = query.endpoint(a, &bands, endpoint)?;
                    let raw = target.retained_endpoint_area(a, &bands, &boundaries)?;
                    assert_eq!(raw.high().item::<f64>()?, 0.0);
                    assert_eq!(raw.low().item::<f64>()?, 0.0);
                    assert!(raw.abs_error().item::<f64>()? > 0.0);
                    Ok(())
                },
            ))?;
        }
        Ok(())
    }

    #[test]
    fn configured_cpu_composed_potential_retains_positive_underflow_water() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        composed_underflow::<CpuRuntime>(&client)?;
        assert_eq!(client.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_composed_potential_retains_positive_underflow_water() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        composed_underflow::<CudaRuntime>(&CudaClient::new(CudaDevice::new(0))?)
    }
}
