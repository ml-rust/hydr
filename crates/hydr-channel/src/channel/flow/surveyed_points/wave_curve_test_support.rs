use super::super::potential_range_tests::frozen::query_profiles as points;
use super::*;
use crate::channel::section::{
    AreaAnchor, AreaDirection, AreaQuery, CrossSectionBatch, CrossSectionDefinition,
    CrossSectionPoint, PrismaticSectionBatch,
};

pub(super) fn sections<R>(client: &R::Client) -> Result<Vec<PrismaticSectionBatch<R>>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let query = points::PROFILES
        .iter()
        .map(|p| (p.id, p.coordinates, p.cap_stage))
        .collect::<Vec<_>>();
    let original: fixtures::Profile = fixtures::ORIGINAL_RECTANGLE;
    let large: fixtures::Profile = fixtures::LARGE_RECTANGLE;
    [
        query,
        vec![(
            "original rectangle",
            original.coordinates,
            original.cap_stage,
        )],
        vec![("large rectangle", large.coordinates, large.cap_stage)],
    ]
    .into_iter()
    .map(|profiles| {
        let definitions = profiles
            .into_iter()
            .map(|(id, coordinates, cap)| CrossSectionDefinition {
                id: id.into(),
                vertical_datum: "Frozen wave reference datum".into(),
                points: coordinates
                    .iter()
                    .map(|point| CrossSectionPoint {
                        station_m: f64::from_bits(point[0]),
                        elevation_m: f64::from_bits(point[1]),
                    })
                    .collect(),
                maximum_stage_m: f64::from_bits(cap),
                profile_provenance: "Independent wave-curve original-coordinate reference".into(),
            })
            .collect();
        Ok(CrossSectionBatch::<R>::new(client.clone(), definitions)?.certify_prismatic()?)
    })
    .collect()
}

pub(super) fn with_case<R, F>(
    client: &R::Client,
    sections: &[PrismaticSectionBatch<R>],
    case: &fixtures::Case,
    strided: bool,
    limit: usize,
    operation: F,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    F: FnOnce(&PotentialContext<'_, R>, PhysicalEndpoint) -> Result,
{
    let (model, owner, source_bits, gravity_bits, chart) = match case.input {
        fixtures::Input::Baseline {
            binding_index,
            original_parameter_endpoint: _,
        } => {
            let binding = &points::CASES[binding_index];
            let chart = &points::QUERIES[binding.query_index];
            (
                0,
                chart.profile_index,
                binding.source_components,
                points::GRAVITY_BITS,
                Some(chart),
            )
        }
        fixtures::Input::SourceRelative {
            profile,
            gravity_bits,
            source,
            delta: _,
        } => {
            let (model, owner) = match profile {
                fixtures::ProfileRef::Query(owner) => (0, owner),
                fixtures::ProfileRef::OriginalRectangle => (1, 0),
                fixtures::ProfileRef::LargeRectangle => (2, 0),
            };
            (model, owner, source, gravity_bits, None)
        }
    };
    let section = &sections[model];
    let geometry = HydraulicGeometry::from_batch(section);
    let owners = Tensor::from_slice(&[i64::try_from(owner)?; 3], &[3], client.device())?;
    let area = expansion::<R>(client, &[[source_bits[0], source_bits[1], 0]; 3], strided)?;
    let before = components(&area)?;
    let mut config = PrismaticRiemannConfig {
        gravity_m_per_s2: f64::from_bits(gravity_bits),
        water_flux_atol_m3_per_s: 1e-12,
        momentum_flux_atol_m4_per_s2: 1e-12,
        flux_rtol: 1e-12,
        max_native_query_points: limit,
        ..PrismaticRiemannConfig::default()
    };
    config.root.atol = 1e-15;
    config.root.rtol = 1e-15;
    config.root.max_iterations = 256;
    let source = section.recover_area_points_tensor(&owners, &area, &config.recovery)?;
    let anchor_area = if let Some(chart) = chart {
        expansion::<R>(
            client,
            &[[
                chart.anchor_source_components[0],
                chart.anchor_source_components[1],
                0,
            ]; 3],
            strided,
        )?
    } else {
        expansion::<R>(client, &[[source_bits[0], source_bits[1], 0]; 3], strided)?
    };
    let anchor = section.recover_area_points_tensor(&owners, &anchor_area, &config.recovery)?;
    let (bands, boundaries, query, endpoint) = hydraulic(error::checked::<R, _>(
        client,
        "frozen wave source graph",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let make_point = |a: &mut error::Arithmetic<'_, R>,
                              components: [u64; 2]|
             -> error::HydraulicResult<
                solvr::common::arithmetic::CertifiedExpansion<R>,
            > {
                Ok(a.expansion_bounds(
                    &Tensor::from_slice(
                        &[f64::from_bits(components[0]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::from_slice(
                        &[f64::from_bits(components[1]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::zeros(&[3], DType::F64, client.device())?,
                )?)
            };
            let (query, endpoint) = if let Some(chart) = chart {
                let scale = make_point(a, chart.scale_components)?;
                let parameter = a.bounds(
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.parameter_bounds[0]); 3],
                        &[3],
                        client.device(),
                    )?,
                    &Tensor::from_slice(
                        &[f64::from_bits(chart.parameter_bounds[1]); 3],
                        &[3],
                        client.device(),
                    )?,
                )?;
                let direction = if chart.add {
                    AreaDirection::Add
                } else {
                    AreaDirection::Subtract
                };
                let query = if chart.anchor == points::AnchorKind::Source {
                    AreaQuery::source_affine(a, &bands, &anchor, direction, &scale, &parameter)?
                } else {
                    let indices = Tensor::from_slice(&[chart.band; 3], &[3], client.device())?;
                    let anchor = match chart.anchor {
                    points::AnchorKind::Zero => AreaAnchor::Zero,
                    points::AnchorKind::BandStart => AreaAnchor::BandStart(indices),
                    points::AnchorKind::BandEnd => AreaAnchor::BandEnd(indices),
                    points::AnchorKind::Cap => AreaAnchor::Cap,
                    points::AnchorKind::Source => return Err(HydraulicError::Numerical { context:
                        "wave chart has inconsistent source anchor dispatch. Check frozen chart metadata".into() }),
                };
                    AreaQuery::band_affine(a, &bands, anchor, direction, &scale, &parameter)?
                };
                let parameter_index = match case.input { fixtures::Input::Baseline { original_parameter_endpoint, .. } => original_parameter_endpoint,
                _ => return Err(HydraulicError::Numerical { context: "baseline wave chart lacks parameter endpoint metadata. Check fixture dispatch".into() }) };
                let endpoint = if (parameter_index == 0) == chart.add {
                    PhysicalEndpoint::Lower
                } else {
                    PhysicalEndpoint::Upper
                };
                (query, endpoint)
            } else {
                let delta =
                    match case.input {
                        fixtures::Input::SourceRelative { delta, .. } => delta,
                        _ => return Err(HydraulicError::Numerical {
                            context:
                                "relative wave chart lacks its retained gap. Check fixture dispatch"
                                    .into(),
                        }),
                    };
                let negative = f64::from_bits(delta[0]) < 0.0
                    || (f64::from_bits(delta[0]) == 0.0 && f64::from_bits(delta[1]) < 0.0);
                let magnitude = if negative {
                    delta.map(|v| (-f64::from_bits(v)).to_bits())
                } else {
                    delta
                };
                let scale = make_point(a, magnitude)?;
                let parameter = a.point(&Tensor::ones(&[3], DType::F64, client.device())?)?;
                let direction = if negative {
                    AreaDirection::Subtract
                } else {
                    AreaDirection::Add
                };
                (
                    AreaQuery::source_affine(a, &bands, &source, direction, &scale, &parameter)?,
                    PhysicalEndpoint::Lower,
                )
            };
            let query = query.prove_support(a, &bands)?;
            Ok((bands, boundaries, query, endpoint))
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
            panels: 1024,
            work: &work,
        },
    ))?;
    operation(&context, endpoint)?;
    assert_eq!(
        before,
        components(&area)?,
        "{}: source input mutation",
        case.id
    );
    assert_eq!(work.snapshot().root_solves, 0);
    assert_eq!(work.snapshot().bracket_solves, 0);
    assert_eq!(work.snapshot().refinements, 0);
    Ok(())
}

pub(super) fn check(actual: [f64; 2], reference: fixtures::Reference, id: &str) {
    let (expected, allowance) = match reference {
        fixtures::Reference::ExactZero => ([0.0; 2], 0.0),
        fixtures::Reference::Local { bounds, allowance } => {
            (bounds.map(f64::from_bits), f64::from_bits(allowance))
        }
    };
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{id}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{id}: {actual:?} excludes {expected:?}"
    );
    let excess = (actual[0] - expected[1])
        .abs()
        .max((actual[1] - expected[0]).abs());
    assert!(
        excess <= allowance,
        "{id}: error={excess:e}, allowance={allowance:e}, actual={actual:?}"
    );
}

pub(super) fn check_value<R>(
    client: &R::Client,
    value: &CurveValue<R>,
    reference: fixtures::Reference,
    id: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(value.available.try_to_vec::<f64>()?, [1.0; 3], "{id}");
    for bounds in [
        &value.physical,
        &super::super::source_tests::support::bounds::<R>(client, &value.retained)?,
    ] {
        for actual in endpoints(bounds)? {
            check(actual, reference, id);
        }
    }
    if matches!(reference, fixtures::Reference::ExactZero) {
        for point in components(&value.retained)? {
            zero(point, id);
        }
    }
    Ok(())
}

pub(super) fn unavailable_value<R: Runtime<DType = DType>>(value: &CurveValue<R>) -> Result {
    assert_eq!(value.available.try_to_vec::<f64>()?, [0.0; 3]);
    for point in components(&value.retained)? {
        zero(point, "unavailable wave placeholder");
    }
    assert_eq!(endpoints(&value.physical)?, [[0.0; 2]; 3]);
    Ok(())
}
