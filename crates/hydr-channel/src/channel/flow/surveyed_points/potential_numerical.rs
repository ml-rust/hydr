use super::potential_nodes::NodeCallbacks;
use super::potential_range::{
    PotentialContext, PotentialValue, from_interval, scatter, zero_value,
};
use super::{analytic_endpoint, potential_analytic};
use crate::channel::{
    flow::surveyed::{
        derivatives::WetDepthPotential,
        equation_ops::scale,
        error::{HydraulicError, HydraulicResult, checked},
        lane_quadrature::{numerical_contributions, simpson_contributions},
    },
    section::ConditionalBandSegments,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, MidpointOps, NextRepresentableOps,
        ScalarOps, SortingOps, TensorOps, TwoSumOps, UnaryOps, UtilityOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
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
        if faces.shape() != [faces.numel()]
            || faces.dtype() != DType::I64
            || !faces.device().is_same(c.device())
        {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "point wet potential face metadata shape={:?}, dtype={:?}, device={}: supply rank-one I64 faces on device={}",
                    faces.shape(),
                    faces.dtype(),
                    faces.device().id(),
                    c.device().id(),
                ),
            });
        }
        let invalid_shape = || HydraulicError::Numerical {
            context: format!(
                "point wet potential shape faces={n}, selected={}, bands={band_count} exceeds exact lane indexing. Split the face batch",
                faces.numel(),
            ),
        };
        let total = n.checked_mul(band_count).ok_or_else(invalid_shape)?;
        i64::try_from(total).map_err(|_| invalid_shape())?;
        let limit = i64::try_from(n).map_err(|_| invalid_shape())?;
        if faces.numel() != 0 {
            let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
            let upper = Tensor::from_slice(&[limit], &[], c.device())?;
            let invalid = c.maximum(&c.lt(faces, &zero)?, &c.ge(faces, &upper)?)?;
            let invalid = c.nonzero(&invalid)?.flatten()?;
            if invalid.numel() != 0 {
                let first = invalid.narrow(0, 0, 1)?;
                let face = c.index_select(faces, 0, &first)?.item::<i64>()?;
                return Err(HydraulicError::Numerical {
                    context: format!(
                        "point wet potential face={face} lies outside 0..{n}. Supply an admitted face",
                    ),
                });
            }
        }
        let count = faces
            .numel()
            .checked_mul(band_count)
            .ok_or_else(invalid_shape)?;
        let exact_range_limit = 1_u128 << f64::MANTISSA_DIGITS;
        if band_count as u128 > exact_range_limit || count as u128 > exact_range_limit {
            return Err(invalid_shape());
        }
        let stride = Tensor::from_slice(
            &[i64::try_from(band_count).map_err(|_| invalid_shape())?],
            &[],
            c.device(),
        )?;
        let first = c.mul(&faces.unsqueeze(1)?, &stride)?;
        let bands = c
            .arange(0.0, band_count as f64, 1.0, DType::I64)?
            .unsqueeze(0)?;
        let rows = c.add(&first, &bands)?.reshape(&[count])?;
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
    let options = CertifiedSimpsonOptions {
        panels: context.evaluation.panels,
        max_points: context.evaluation.config.max_native_query_points,
    };
    let work = context.evaluation.work;
    let result = if let Some(depth) = &depth {
        numerical_contributions(
            c,
            &left,
            &right,
            options,
            work,
            |a, owner, parameter| {
                WetDepthPotential::from_geometry(a, depth, owner, parameter, gravity)?.value(a)
            },
            |a, owner, parameter| {
                let derivative =
                    WetDepthPotential::from_geometry(a, depth, owner, parameter, gravity)?
                        .fourth_derivative(a)?;
                Ok(c.maximum(&c.abs(derivative.lower())?, &c.abs(derivative.upper())?)?)
            },
        )?
    } else {
        let mut callbacks = NodeCallbacks::new(&selected, gravity);
        simpson_contributions(c, &left, &right, options, work, &mut callbacks)?
    };
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::{PrismaticRiemannConfig, error, work::WorkLedger};
    use crate::channel::flow::surveyed_points::potential_increment_fixtures as frozen;
    use crate::channel::flow::surveyed_points::potential_range::PotentialEvaluation;
    use crate::channel::flow::surveyed_points::source_fixture_support::{
        Client, Result, components, endpoints, expansion, hydraulic,
    };
    use crate::channel::section::{
        AreaDirection, AreaQuery, CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint,
        HydraulicGeometry, PrismaticRecoveryConfig,
    };

    fn face_band_order<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
    {
        let profile = frozen::query_profiles::PROFILES[2];
        let section = CrossSectionBatch::<R>::new(
            client.clone(),
            vec![CrossSectionDefinition {
                id: profile.id.into(),
                vertical_datum: "Frozen potential datum".into(),
                points: profile
                    .coordinates
                    .iter()
                    .map(|point| CrossSectionPoint {
                        station_m: f64::from_bits(point[0]),
                        elevation_m: f64::from_bits(point[1]),
                    })
                    .collect(),
                maximum_stage_m: f64::from_bits(profile.cap_stage),
                profile_provenance: "Independent original-coordinate potential reference".into(),
            }],
        )?
        .certify_prismatic()?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let cases = [
            frozen::query_profiles::CASES[13],
            frozen::query_profiles::CASES[17],
            frozen::query_profiles::CASES[18],
        ];
        assert_eq!(
            cases.map(|case| case.id),
            [
                "three_band_broad_reverse__physical_anchor",
                "first_knot_from_second_source",
                "second_knot_from_first_source",
            ]
        );
        let owners = Tensor::from_slice(&[0_i64; 3], &[3], client.device())?;
        let source_components =
            cases.map(|case| [case.source_components[0], case.source_components[1], 0]);
        let source_area = expansion::<R>(&client, &source_components, true)?;
        let target_components = [
            [1.0_f64.to_bits(), 0, 0],
            [3.0_f64.to_bits(), (-2.0_f64.powi(-80)).to_bits(), 0],
            [7.5_f64.to_bits(), 2.0_f64.powi(-80).to_bits(), 0],
        ];
        let target_area = expansion::<R>(&client, &target_components, true)?;
        let before_source = components(&source_area)?;
        let before_target = components(&target_area)?;
        let recovery = PrismaticRecoveryConfig::default();
        let source = section.recover_area_points_tensor(&owners, &source_area, &recovery)?;
        let target = section.recover_area_points_tensor(&owners, &target_area, &recovery)?;
        let (bands, boundaries, query) = hydraulic(error::checked::<R, _>(
            &client,
            "ordered potential test binding",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let boundaries = geometry.boundary_areas(a)?.gather(a, &owners)?;
                let zero = Tensor::zeros(&[3], a.dtype(), client.device())?;
                let scale = a.expansion_point(&zero)?;
                let parameter = a.point(&zero)?;
                let query = AreaQuery::source_affine(
                    a,
                    &bands,
                    &target,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                Ok((bands, boundaries, query))
            },
        ))?;
        assert_eq!(bands.band_count, 3);
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: 1.0,
            ..PrismaticRiemannConfig::default()
        };
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
        let segments = hydraulic(error::checked::<R, _>(
            &client,
            "ordered potential test segments",
            |a| ConditionalBandSegments::prepare(a, &bands, &source, &query, &boundaries),
        ))?;
        let work_before_rejection = work.snapshot();
        for rows in [
            Tensor::from_slice(&[6_148_914_691_236_517_206_i64], &[1], client.device())?,
            Tensor::from_slice(&[-1_i64], &[1], client.device())?,
            Tensor::from_slice(&[3_i64], &[1], client.device())?,
            Tensor::from_slice(&[0_i64], &[1, 1], client.device())?,
            Tensor::from_slice(&[0_i64], &[], client.device())?,
            Tensor::from_slice(&[0.0_f64], &[1], client.device())?,
        ] {
            assert!(matches!(
                wet_contributions(&context, &segments, &rows),
                Err(error::HydraulicError::Numerical { .. })
            ));
            assert_eq!(work_before_rejection, work.snapshot());
            assert_eq!(before_source, components(&source_area)?);
            assert_eq!(before_target, components(&target_area)?);
        }
        let references = [
            cases[0].endpoints[1].increment,
            cases[1].endpoints[0].increment,
            cases[2].endpoints[0].increment,
        ];
        let mut singleton = Vec::new();
        for (face, reference) in references.iter().enumerate() {
            let rows = Tensor::from_slice(&[i64::try_from(face)?], &[1], client.device())?;
            let value = hydraulic(wet_contributions(&context, &segments, &rows))?;
            let physical = endpoints(&value.physical)?[0];
            check_reference(physical, *reference, "singleton physical");
            singleton.push(physical);
        }
        for layout in [&[2_i64, 0, 1][..], &[1, 2, 0], &[2, 0], &[1, 1, 0, 2]] {
            let rows = Tensor::from_slice(layout, &[layout.len()], client.device())?;
            let value = hydraulic(wet_contributions(&context, &segments, &rows))?;
            let physical = endpoints(&value.physical)?;
            let retained = hydraulic(error::checked::<R, _>(
                &client,
                "ordered potential test retained bounds",
                |a| Ok(a.expansion_interval(&value.retained)?),
            ))?;
            assert_eq!(physical.len(), layout.len());
            let retained = endpoints(&retained)?;
            for (position, face) in layout.iter().enumerate() {
                let face = usize::try_from(*face)?;
                let label = format!("layout={layout:?}, position={position}, face={face}");
                check_reference(physical[position], references[face], &label);
                check_reference(retained[position], references[face], &label);
                assert!(
                    physical[position][0] <= singleton[face][1]
                        && singleton[face][0] <= physical[position][1],
                    "{label}"
                );
            }
        }
        assert_eq!(before_source, components(&source_area)?);
        assert_eq!(before_target, components(&target_area)?);
        assert_eq!(work.snapshot().root_solves, 0);
        assert_eq!(work.snapshot().bracket_solves, 0);
        Ok(())
    }

    fn check_reference(actual: [f64; 2], reference: frozen::Reference, label: &str) {
        let expected = reference.bounds.map(f64::from_bits);
        assert!(
            actual[0] <= expected[0] && expected[1] <= actual[1],
            "{label}: actual={actual:?}, independent={expected:?}"
        );
        let excess = (actual[0] - expected[1])
            .abs()
            .max((actual[1] - expected[0]).abs());
        assert!(
            excess <= f64::from_bits(reference.native_allowance),
            "{label}: excess={excess:e}, allowance={:e}",
            f64::from_bits(reference.native_allowance)
        );
    }

    #[test]
    fn configured_cpu_potential_numerical_preserves_face_and_band_order() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        face_band_order::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_potential_numerical_preserves_face_and_band_order() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        face_band_order::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
