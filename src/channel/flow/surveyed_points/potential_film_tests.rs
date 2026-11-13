use super::*;

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

pub(super) fn run<R>(client: &R::Client) -> Result
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
    let source =
        section.recover_area_points_tensor(&owners, &area, &PrismaticRecoveryConfig::default())?;
    let config = config(inputs.gravity_bits);
    let work = WorkLedger::new(&config);
    let (bands, boundaries, epsilon) = hydraulic(error::checked::<R, _>(
        client,
        "original film invariant",
        |a| {
            let bands = geometry.gather(a, &owners)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
            let velocity = a.expansion_div_positive(&discharge, &area)?;
            let velocity_points = components(&velocity).map_err(|e| HydraulicError::Numerical {
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
            let four = a.expansion_point(&Tensor::from_slice(&[4.0; 2], &[2], client.device())?)?;
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
            let residual = hydraulic(error::checked::<R, _>(client, "film star residual", |a| {
                let two =
                    a.expansion_point(&Tensor::from_slice(&[2.0; 2], &[2], client.device())?)?;
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
            assert!(bracket.zero_anchor && bracket.add && bracket.strict_opposite_residual_signs);
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
                            value
                                .analytic
                                .as_ref()
                                .ok_or_else(|| HydraulicError::Numerical {
                                    context:
                                        "film endpoint lacks analytic metadata. Check dispatch"
                                            .into(),
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
                let area = hydraulic(error::checked::<R, _>(client, "film directed area", |a| {
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
