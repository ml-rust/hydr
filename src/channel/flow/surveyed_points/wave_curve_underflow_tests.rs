use super::*;
use crate::channel::section::source_difference;

#[path = "wave_curve_scaling_fixture.rs"]
mod scaling;

fn contraction_underflow<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let sections = support::sections::<R>(&client)?;
    let case = &fixtures::CASES[fixtures::CONTRACTION_UNDERFLOW.case_index];
    support::with_case::<R, _>(
        &client,
        &sections,
        case,
        true,
        1 << 24,
        |context, endpoint| {
            let fixtures::Input::SourceRelative { source, delta, .. } = case.input else {
                return Err(
                    "Contraction control requires its original source-relative input".into(),
                );
            };
            assert_eq!(delta, [1, 0], "{}: original minimum-subnormal gap", case.id);
            let expected_delta = [[delta[0], delta[1], 0]; 3];
            let source_before = components(context.source.requested_areas_m2())?;
            let scale_before = components(&context.query.query.scale)?;
            let parameter_before = endpoints(&context.query.query.parameter)?;
            let owners_before = context.query.query.owners.try_to_vec::<i64>()?;
            let source_owners_before = context.source.owners().try_to_vec::<i64>()?;
            assert_eq!(
                source_before,
                [[source[0], source[1], 0]; 3],
                "{}: original source",
                case.id
            );
            assert_eq!(
                scale_before, expected_delta,
                "{}: exact query scale",
                case.id
            );
            assert_eq!(
                parameter_before, [[1.0; 2]; 3],
                "{}: unit map parameter",
                case.id
            );
            let (differences, endpoint_difference, product) = hydraulic(error::checked::<R, _>(
                &client,
                "frozen contraction underflow stages",
                |a| {
                    let query = context.query.endpoint(a, context.bands, endpoint)?;
                    let differences = source_difference(
                        a,
                        context.bands,
                        context.query,
                        context.source,
                        context.boundaries,
                    )?;
                    let endpoint_difference = source_difference(
                        a,
                        context.bands,
                        &query,
                        context.source,
                        context.boundaries,
                    )?;
                    let parameter = a.expansion_point(context.query.query.parameter.lower())?;
                    let product = a.expansion_mul(&context.query.query.scale, &parameter)?;
                    Ok((differences, endpoint_difference, product))
                },
            ))?;
            for (label, value) in [
                ("minimum-subnormal times one", &product),
                ("stored map difference lower", &differences.lower),
                ("stored map difference upper", &differences.upper),
                ("endpoint difference lower", &endpoint_difference.lower),
                ("endpoint difference upper", &endpoint_difference.upper),
            ] {
                assert_eq!(components(value)?, expected_delta, "{}: {label}", case.id);
            }
            let classification = hydraulic(error::checked::<R, _>(
                &client,
                "underflow shock classification",
                |a| wave_curve::classify(a, context.source.dry_mask(), &endpoint_difference.lower),
            ))?;
            assert_eq!(
                classification.shock.try_to_vec::<f64>()?,
                [1.0; 3],
                "{}: definite shock",
                case.id
            );
            for (label, mask) in [
                ("rarefaction", &classification.rarefaction),
                ("equality", &classification.equality),
                ("uncertain", &classification.uncertain),
                ("dry source", &classification.dry_source),
            ] {
                assert_eq!(mask.try_to_vec::<f64>()?, [0.0; 3], "{}: {label}", case.id);
            }
            let value = hydraulic(wave_curve::endpoint(context, endpoint))?;
            check_endpoint::<R>(&client, &value, case.expected, case.id)?;
            assert_eq!(
                components(context.source.requested_areas_m2())?,
                source_before,
                "{}: source snapshot",
                case.id
            );
            assert_eq!(
                components(&context.query.query.scale)?,
                scale_before,
                "{}: scale snapshot",
                case.id
            );
            assert_eq!(
                endpoints(&context.query.query.parameter)?,
                parameter_before,
                "{}: parameter snapshot",
                case.id
            );
            assert_eq!(
                context.query.query.owners.try_to_vec::<i64>()?,
                owners_before,
                "{}: owner snapshot",
                case.id
            );
            assert_eq!(
                context.source.owners().try_to_vec::<i64>()?,
                source_owners_before,
                "{}: source owner snapshot",
                case.id
            );
            let stored_after = hydraulic(error::checked::<R, _>(
                &client,
                "underflow stored map snapshot",
                |a| {
                    source_difference(
                        a,
                        context.bands,
                        context.query,
                        context.source,
                        context.boundaries,
                    )
                },
            ))?;
            for (label, value) in [
                ("stored map lower snapshot", &stored_after.lower),
                ("stored map upper snapshot", &stored_after.upper),
            ] {
                assert_eq!(components(value)?, expected_delta, "{}: {label}", case.id);
            }
            Ok(())
        },
    )
}

fn large_area<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    use crate::channel::section::{CrossSectionBatch, CrossSectionDefinition, CrossSectionPoint};
    let definition = CrossSectionDefinition {
        id: "large area and small gravity".into(),
        vertical_datum: "Independent scaling reference datum".into(),
        points: scaling::COORDINATES
            .iter()
            .map(|point| CrossSectionPoint {
                station_m: f64::from_bits(point[0]),
                elevation_m: f64::from_bits(point[1]),
            })
            .collect(),
        maximum_stage_m: f64::from_bits(scaling::CAP_STAGE),
        profile_provenance: "Frozen independent side-curve scaling addendum".into(),
    };
    let section =
        CrossSectionBatch::<R>::new(client.clone(), vec![definition])?.certify_prismatic()?;
    let case = fixtures::Case {
        id: "large area and small gravity compression",
        input: fixtures::Input::SourceRelative {
            profile: fixtures::ProfileRef::Query(0),
            gravity_bits: scaling::GRAVITY,
            source: scaling::SOURCE,
            delta: scaling::DELTA,
        },
        expected: fixtures::Expected::Shock {
            curve: fixtures::Reference::Local {
                bounds: scaling::CURVE,
                allowance: scaling::CURVE_ALLOWANCE,
            },
            offset: fixtures::Reference::Local {
                bounds: scaling::OFFSET,
                allowance: scaling::OFFSET_ALLOWANCE,
            },
        },
    };
    support::with_case::<R, _>(
        &client,
        &[section],
        &case,
        true,
        1 << 24,
        |context, endpoint| {
            let value = hydraulic(wave_curve::endpoint(context, endpoint))?;
            check_endpoint::<R>(&client, &value, case.expected, case.id)
        },
    )
}

#[test]
fn configured_cpu_contraction_underflow_preserves_exact_map_and_shock_curve() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    contraction_underflow::<CpuRuntime>(client)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_contraction_underflow_preserves_exact_map_and_shock_curve() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    contraction_underflow::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}

#[test]
fn configured_cpu_small_gravity_large_area_side_curve_preserves_ordering() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client = CpuClient::new(CpuDevice::new())
        .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
    large_area::<CpuRuntime>(client)
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_small_gravity_large_area_side_curve_preserves_ordering() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    large_area::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
