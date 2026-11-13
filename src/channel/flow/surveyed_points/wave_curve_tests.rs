use super::potential_range::{PotentialContext, PotentialEvaluation};
use super::source_tests::support::{
    Client, Result, components, endpoints, expansion, hydraulic, zero,
};
use super::wave_curve;
use super::wave_curve_types::{CurveValue, EndpointCurve};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig, PrismaticRiemannRejection,
        error::{self, HydraulicError},
        work::WorkLedger,
    },
    section::{HydraulicGeometry, PhysicalEndpoint},
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::ArithmeticError, integrate::CertifiedSimpsonAlgorithms};

#[path = "wave_curve_conditional_tests.rs"]
mod conditional;
#[path = "wave_curve_fixtures/mod.rs"]
mod fixtures;
#[path = "wave_curve_mixed_tests.rs"]
mod mixed;
#[path = "wave_curve_test_support.rs"]
mod support;
#[path = "wave_curve_underflow_tests.rs"]
mod underflow;

fn check_endpoint<R>(
    client: &R::Client,
    value: &EndpointCurve<R>,
    expected: fixtures::Expected,
    id: &str,
) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let (shock, rare, equal, dry) = match expected {
        fixtures::Expected::Dry => {
            support::unavailable_value(&value.curve)?;
            (0.0, 0.0, 0.0, 1.0)
        }
        fixtures::Expected::Equality { limit } => {
            support::check_value::<R>(client, &value.curve, fixtures::Reference::ExactZero, id)?;
            support::check_value::<R>(client, &value.equality_limit, limit, id)?;
            (0.0, 0.0, 1.0, 0.0)
        }
        fixtures::Expected::Rare { curve } => {
            support::check_value::<R>(client, &value.curve, curve, id)?;
            for bounds in endpoints(&value.curve.physical)? {
                assert!(bounds[1] <= 0.0, "{id}: {bounds:?}");
            }
            (0.0, 1.0, 0.0, 0.0)
        }
        fixtures::Expected::Shock { curve, offset } => {
            support::check_value::<R>(client, &value.curve, curve, id)?;
            support::check_value::<R>(client, &value.actual_shock.value, offset, id)?;
            for bounds in endpoints(&value.curve.physical)? {
                assert!(bounds[0] > 0.0, "{id}: {bounds:?}");
            }
            let domain = &value.actual_shock.domain;
            for bounds in endpoints(&domain.source_area)?
                .into_iter()
                .chain(endpoints(&domain.target_area)?)
            {
                assert!(
                    bounds[0] > 0.0 && bounds[0] <= bounds[1],
                    "{id}: {bounds:?}"
                );
            }
            for bounds in endpoints(&domain.separation)? {
                assert!(bounds[0] > 0.0, "{id}: {bounds:?}");
            }
            for bounds in endpoints(&domain.contraction)? {
                assert!(bounds[0] >= 0.0 && bounds[1] <= 1.0, "{id}: {bounds:?}");
            }
            (1.0, 0.0, 0.0, 0.0)
        }
        fixtures::Expected::NumericalRange => {
            return Err("Numerical range case returned a wave certificate".into());
        }
    };
    assert_eq!(
        value.classification.shock.try_to_vec::<f64>()?,
        [shock; 3],
        "{id}"
    );
    assert_eq!(
        value.classification.rarefaction.try_to_vec::<f64>()?,
        [rare; 3],
        "{id}"
    );
    assert_eq!(
        value.classification.equality.try_to_vec::<f64>()?,
        [equal; 3],
        "{id}"
    );
    assert_eq!(
        value.classification.dry_source.try_to_vec::<f64>()?,
        [dry; 3],
        "{id}"
    );
    assert_eq!(
        value.classification.uncertain.try_to_vec::<f64>()?,
        [0.0; 3],
        "{id}"
    );
    support::unavailable_value(&value.conditional_shock.value)?;
    if shock == 0.0 {
        support::unavailable_value(&value.actual_shock.value)?;
    }
    if equal == 0.0 {
        support::unavailable_value(&value.equality_limit)?;
    }
    Ok(())
}

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R> + CertifiedSimpsonAlgorithms<R>,
{
    let sections = support::sections::<R>(&client)?;
    for strided in [false, true] {
        for case in fixtures::CASES {
            support::with_case::<R, _>(
                &client,
                &sections,
                case,
                strided,
                1 << 24,
                |context, endpoint| {
                    let before = context.evaluation.work.snapshot();
                    let result = wave_curve::endpoint(context, endpoint);
                    if matches!(case.expected, fixtures::Expected::NumericalRange) {
                        let error = result
                            .err()
                            .ok_or("Numerical range case returned a wave certificate")?;
                        assert!(
                            matches!(
                                error,
                                HydraulicError::Arithmetic(ArithmeticError::Range { .. })
                            ),
                            "{}: expected arithmetic range, actual={error:?}",
                            case.id
                        );
                        return Ok(());
                    }
                    let value = hydraulic(result)?;
                    check_endpoint::<R>(&client, &value, case.expected, case.id)?;
                    if matches!(
                        case.expected,
                        fixtures::Expected::Dry
                            | fixtures::Expected::Equality { .. }
                            | fixtures::Expected::Shock { .. }
                    ) {
                        assert_eq!(
                            context.evaluation.work.snapshot(),
                            before,
                            "{}: unrequested quadrature",
                            case.id
                        );
                    }
                    if matches!(case.expected, fixtures::Expected::Dry) {
                        // Dry-source availability remains separate from a lazily requested target potential.
                        if let fixtures::Input::Baseline {
                            binding_index,
                            original_parameter_endpoint,
                        } = case.input
                        {
                            let reference =
                                super::potential_range_tests::frozen::query_profiles::CASES
                                    [binding_index]
                                    .endpoints[original_parameter_endpoint]
                                    .potential;
                            let target = hydraulic(super::potential_absolute::endpoint_potential(
                                context, endpoint,
                            ))?;
                            let expected = fixtures::Reference::Local {
                                bounds: reference.bounds,
                                allowance: reference.native_allowance,
                            };
                            for bounds in endpoints(&target.physical)? {
                                support::check(bounds, expected, case.id);
                            }
                        }
                    }
                    Ok(())
                },
            )
            .map_err(|error| format!("{} (strided={strided}): {error}", case.id))?;
        }
    }
    for index in fixtures::SOURCE_CROSSING {
        let case = &fixtures::CASES[index];
        support::with_case::<R, _>(&client, &sections, case, true, 1 << 24, |context, _| {
            let range = hydraulic(wave_curve::range(context))?;
            let lower_case = &fixtures::CASES[fixtures::SOURCE_CROSSING[0]];
            let upper_case = &fixtures::CASES[fixtures::SOURCE_CROSSING[1]];
            check_endpoint::<R>(&client, &range.lower, lower_case.expected, lower_case.id)?;
            check_endpoint::<R>(&client, &range.upper, upper_case.expected, upper_case.id)?;
            assert_eq!(range.curve.available.try_to_vec::<f64>()?, [1.0; 3]);
            let physical = endpoints(&range.curve.physical)?;
            let retained =
                super::source_tests::support::bounds::<R>(&client, &range.curve.retained)?;
            let retained = endpoints(&retained)?;
            let lower = endpoints(&range.lower.curve.physical)?;
            let upper = endpoints(&range.upper.curve.physical)?;
            for row in 0..3 {
                assert_eq!(physical[row], [lower[row][0], upper[row][1]]);
                assert!(
                    retained[row][0] <= physical[row][0] && physical[row][1] <= retained[row][1]
                );
            }
            Ok(())
        })?;
    }
    let rare = fixtures::CASES
        .iter()
        .find(|case| {
            matches!(
                case.input,
                fixtures::Input::Baseline {
                    binding_index: 13,
                    ..
                }
            )
        })
        .ok_or("Missing frozen multiband rarefaction")?;
    support::with_case::<R, _>(&client, &sections, rare, true, 0, |context, endpoint| {
        let result = wave_curve::endpoint(context, endpoint);
        assert!(matches!(
            result,
            Err(HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                ..
            })
        ));
        assert_eq!(context.evaluation.work.snapshot(), Default::default());
        Ok(())
    })?;
    conditional::controls::<R>(&client, &sections)
}

#[test]
fn configured_cpu_side_curves_preserve_retained_offsets_and_physical_branches() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    matrix::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_side_curves_preserve_retained_offsets_and_physical_branches() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
