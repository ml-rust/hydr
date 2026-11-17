use super::*;
use crate::channel::section::{PrismaticSectionBatch, pressure_endpoint};

pub(super) fn controls<R>(client: &R::Client, sections: &[PrismaticSectionBatch<R>]) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let control: fixtures::UncertainSign = fixtures::UNCERTAIN_SIGN;
    let original = &fixtures::CASES[control.case_index];
    support::with_case::<R, _>(
        client,
        sections,
        original,
        true,
        1 << 24,
        |context, endpoint| {
            let (classification, curve, offset, delta) = hydraulic(error::checked::<R, _>(
                client,
                "frozen uncertain sign wave certificate",
                |a| {
                    let query = context.query.endpoint(a, context.bands, endpoint)?;
                    let target =
                        query.retained_endpoint_area(a, context.bands, context.boundaries)?;
                    let mut pressure = pressure_endpoint(
                        a,
                        context.geometry,
                        context.bands,
                        &query,
                        context.source,
                        context.boundaries,
                        context.evaluation.config.gravity_m_per_s2,
                    )?;
                    let center = a.expansion_bounds(
                        &Tensor::from_slice(
                            &[f64::from_bits(control.delta_center[0]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::from_slice(
                            &[f64::from_bits(control.delta_center[1]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::zeros(&[3], DType::F64, client.device())?,
                    )?;
                    let radius = a.expansion_bounds(
                        &Tensor::from_slice(
                            &[f64::from_bits(control.delta_radius[0]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::from_slice(
                            &[f64::from_bits(control.delta_radius[1]); 3],
                            &[3],
                            client.device(),
                        )?,
                        &Tensor::zeros(&[3], DType::F64, client.device())?,
                    )?;
                    let radius = a.expansion_interval(&radius)?;
                    // Only this existing separation certificate widens. Physical source and target inputs stay unchanged.
                    let widened =
                        a.expansion_bounds(center.high(), center.low(), radius.upper())?;
                    let old = a.expansion_interval(&pressure.delta_area)?;
                    let widened_bounds = a.expansion_interval(&widened)?;
                    let old = endpoints(&old).map_err(|e| HydraulicError::Numerical {
                        context: e.to_string(),
                    })?;
                    let broad =
                        endpoints(&widened_bounds).map_err(|e| HydraulicError::Numerical {
                            context: e.to_string(),
                        })?;
                    for (old, broad) in old.into_iter().zip(broad) {
                        assert!(broad[0] <= old[0] && old[1] <= broad[1]);
                        assert!(broad[0] < 0.0 && broad[1] > 0.0);
                    }
                    pressure.delta_area = widened;
                    let classification =
                        wave_curve::classify(a, context.source.dry_mask(), &pressure.delta_area)?;
                    let (curve, offset) = super::super::shock_points::conditional(
                        a,
                        context.source.requested_areas_m2(),
                        &target,
                        &pressure,
                    )?;
                    Ok((classification, curve, offset, pressure.delta_area))
                },
            ))?;
            assert_eq!(classification.uncertain.try_to_vec::<f64>()?, [1.0; 3]);
            if control.overlap_proves_neither_equality_nor_shock {
                assert_eq!(classification.equality.try_to_vec::<f64>()?, [0.0; 3]);
                assert_eq!(classification.shock.try_to_vec::<f64>()?, [0.0; 3]);
            }
            if control.actual_shock_offset_unavailable {
                assert_eq!(classification.shock.try_to_vec::<f64>()?, [0.0; 3]);
            }
            assert_eq!(
                components(&delta)?,
                components(&offset.domain.retained_separation)?
            );
            let expected = match original.expected {
                fixtures::Expected::Shock {
                    curve: fixtures::Reference::Local { bounds, .. },
                    ..
                } => bounds.map(f64::from_bits),
                _ => {
                    return Err(
                        "Frozen uncertain sign control lacks its physical shock reference".into(),
                    );
                }
            };
            for bounds in endpoints(&curve.physical)? {
                assert!(bounds[0] <= 0.0 && bounds[1] >= expected[1]);
                assert!(bounds[0] <= expected[0]);
            }
            for bounds in endpoints(&offset.domain.contraction)? {
                assert!(bounds[0] >= 0.0 && bounds[1] <= 1.0);
            }
            for bounds in endpoints(&offset.domain.separation)? {
                assert_eq!(bounds[0], 0.0);
                assert!(bounds[1] > 0.0);
            }
            assert_eq!(context.evaluation.work.snapshot(), Default::default());
            Ok(())
        },
    )?;
    let underflow: fixtures::ContractionUnderflow = fixtures::CONTRACTION_UNDERFLOW;
    support::with_case::<R, _>(
        client,
        sections,
        &fixtures::CASES[underflow.case_index],
        true,
        1 << 24,
        |context, endpoint| {
            let result = hydraulic(wave_curve::endpoint(context, endpoint))?;
            check_endpoint::<R>(
                client,
                &result,
                fixtures::CASES[underflow.case_index].expected,
                fixtures::CASES[underflow.case_index].id,
            )?;
            if underflow.nonzero_physical_truth {
                for bounds in endpoints(&result.actual_shock.domain.separation)? {
                    assert!(bounds[0] > 0.0);
                }
                for bounds in endpoints(&result.curve.physical)? {
                    assert!(bounds[0] > 0.0);
                }
            }
            let contraction = hydraulic(error::checked::<R, _>(
                client,
                "contraction underflow certificate",
                |a| Ok(a.expansion_from_interval(&result.actual_shock.domain.contraction)?),
            ))?;
            if underflow.zero_center_requires_nonzero_radius {
                for point in components(&contraction)? {
                    let [high, low, radius] = point.map(f64::from_bits);
                    if high == 0.0 && low == 0.0 {
                        assert!(radius > 0.0);
                    }
                }
            }
            assert_eq!(context.evaluation.work.snapshot(), Default::default());
            Ok(())
        },
    )
}
