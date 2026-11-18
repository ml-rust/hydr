use crate::channel::flow::surveyed::{
    analytic_bottom,
    error::{Arithmetic, HydraulicResult},
};
use crate::channel::section::GatheredBands;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

use super::analytic_bottom_arithmetic::{AnalyticOperands, evaluate};

/// Only eligible rows contain available analytic source certificates.
pub(crate) struct AnalyticSource<R: Runtime<DType = DType>> {
    pub(crate) eligible: Tensor<R>,
    pub(crate) rectangular: Tensor<R>,
    pub(crate) pointed: Tensor<R>,
    /// Ineligible rows contain unavailable zero placeholders.
    pub(crate) potential: CertifiedExpansion<R>,
    /// Ineligible rows contain unavailable zero placeholders.
    pub(crate) sound: CertifiedExpansion<R>,
    /// Retains the pre-root radicand relation, including its complete arithmetic radius.
    pub(crate) sound_squared: CertifiedExpansion<R>,
}

/// Use captured source area and its complete recovered-depth certificate.
/// Gravity comes from the checked Riemann config. Noneligible rows require the complete general branch.
pub(crate) fn source<R>(
    a: &mut Arithmetic<'_, R>,
    bands: &GatheredBands<R>,
    exact_area: &CertifiedExpansion<R>,
    complete_depth: &Interval<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticSource<R>>
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
    let c = a.client();
    let zero = Tensor::zeros(exact_area.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(exact_area.high().shape(), DType::F64, c.device())?;
    let column = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
        Ok(value.narrow(1, 0, 1)?.squeeze(Some(1)))
    };
    let width = a.bounds(
        &column(bands.start_width.lower())?,
        &column(bands.start_width.upper())?,
    )?;
    let slope = a.bounds(&column(bands.slope.lower())?, &column(bands.slope.upper())?)?;
    let end_depth = a.bounds(
        &column(bands.end_depth.lower())?,
        &column(bands.end_depth.upper())?,
    )?;
    let (rectangular, pointed) = analytic_bottom::kinds(
        a,
        &column(&bands.valid)?,
        &column(&bands.pointed)?,
        &width,
        &slope,
    )?;
    let end_area = a.expansion_point(&column(bands.end_area.lower())?)?;
    let difference = a.expansion_sub(exact_area, &end_area)?;
    let difference = a.expansion_interval(&difference)?;
    let supported = c.le(difference.upper(), &zero)?;
    let rectangular = c.mul(&rectangular, &supported)?;
    let pointed = c.mul(&pointed, &supported)?;
    let eligible = c.maximum(&rectangular, &pointed)?;

    evaluate(
        a,
        AnalyticOperands {
            eligible,
            rectangular,
            pointed,
            exact_area,
            complete_depth,
            width: &width,
            depth_identity: Some((&slope, &end_depth)),
            zero: &zero,
            one: &one,
            gravity,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::super::source_potential_fixtures::{CASES, Case, Reference, SideReference};
    use super::super::source_tests::support::{
        self, Client, Result, batch, bounds, components, endpoints, expansion, hydraulic, zero,
    };
    use super::{self as analytic_bottom, AnalyticSource};
    use crate::channel::flow::surveyed::error;
    use crate::channel::section::{HydraulicGeometry, PrismaticRecoveryConfig};
    use numr::dtype::DType;
    use numr::ops::BinaryOps;
    use numr::runtime::{Device, Runtime, RuntimeClient};
    use numr::tensor::Tensor;
    use std::collections::BTreeMap;

    #[derive(Debug, PartialEq)]
    struct Snapshot {
        eligible: Vec<f64>,
        potential: Vec<[u64; 3]>,
        sound: Vec<[u64; 3]>,
    }

    fn snapshot<R: Runtime<DType = DType>>(value: &AnalyticSource<R>) -> Result<Snapshot> {
        Ok(Snapshot {
            eligible: value.eligible.try_to_vec::<f64>()?,
            potential: components(&value.potential)?,
            sound: components(&value.sound)?,
        })
    }

    fn check(actual: [f64; 2], reference: Reference, point: [u64; 3], label: &str) {
        let expected = reference.bounds.map(f64::from_bits);
        assert!(
            actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
            "{label}: {actual:?}"
        );
        assert!(
            actual[0] <= expected[0] && expected[1] <= actual[1],
            "{label}: {actual:?} excludes {expected:?}"
        );
        let error = (actual[0] - expected[1])
            .abs()
            .max((actual[1] - expected[0]).abs());
        assert!(
            error <= f64::from_bits(reference.native_tolerance),
            "{label}: error={error}"
        );
        if reference.exact_zero {
            zero(point, label);
            assert_eq!(actual, [0.0, 0.0], "{label}");
        }
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        assert_eq!(CASES.len(), 70);
        let section = batch::<R>(&client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        let definitions = section.definitions().to_vec();
        let mut groups: BTreeMap<u64, Vec<&Case>> = BTreeMap::new();
        for case in CASES.iter().rev() {
            groups.entry(case.gravity_bits).or_default().push(case);
        }
        let mut exercised = 0;
        for (gravity_bits, cases) in groups {
            let sources = cases
                .iter()
                .map(|case| {
                    support::cases()
                        .get(case.source_index)
                        .ok_or("Missing frozen source index")
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let owners_host = sources
                .iter()
                .map(|source| i64::try_from(source.owner))
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let mut contiguous = None;
            for strided in [false, true] {
                let owners = if strided {
                    let storage: Vec<_> = owners_host
                        .iter()
                        .flat_map(|&owner| [i64::MAX, owner])
                        .collect();
                    let view = Tensor::from_slice(&storage, &[sources.len(), 2], client.device())?
                        .narrow(1, 1, 1)?
                        .squeeze(Some(1));
                    assert!(view.offset() > 0);
                    if sources.len() > 1 {
                        assert!(!view.is_contiguous());
                    }
                    view
                } else {
                    Tensor::from_slice(&owners_host, &[sources.len()], client.device())?
                };
                let inputs = (0..4)
                    .map(|field| {
                        expansion::<R>(
                            &client,
                            &sources
                                .iter()
                                .map(|source| source.inputs[field])
                                .collect::<Vec<_>>(),
                            strided,
                        )
                    })
                    .collect::<Result<Vec<_>>>()?;
                let captured = hydraulic(section.capture_riemann_points(
                    &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3],
                ))?;
                let config = PrismaticRecoveryConfig::default();
                let left_geometry = section.recover_area_points_tensor(
                    &captured.owners,
                    &captured.left_area,
                    &config,
                )?;
                let right_geometry = section.recover_area_points_tensor(
                    &captured.owners,
                    &captured.right_area,
                    &config,
                )?;
                let calculate = || {
                    hydraulic(error::checked::<R, _>(
                        &client,
                        "analytic source potential matrix",
                        |a| {
                            let bands = geometry.gather(a, &captured.owners)?;
                            let left = analytic_bottom::source(
                                a,
                                &bands,
                                &captured.left_area,
                                left_geometry.depth_m(),
                                f64::from_bits(gravity_bits),
                            )?;
                            let right = analytic_bottom::source(
                                a,
                                &bands,
                                &captured.right_area,
                                right_geometry.depth_m(),
                                f64::from_bits(gravity_bits),
                            )?;
                            Ok([left, right])
                        },
                    ))
                };
                let result = calculate()?;
                let snapshots = result.iter().map(snapshot).collect::<Result<Vec<_>>>()?;
                for (side, value) in result.iter().enumerate() {
                    let phi_bounds = endpoints(&bounds::<R>(&client, &value.potential)?)?;
                    let sound_bounds = endpoints(&bounds::<R>(&client, &value.sound)?)?;
                    let actual = &snapshots[side];
                    for (row, case) in cases.iter().enumerate() {
                        let reference: SideReference = case.sides[side];
                        let label = format!("{}, side={side}, strided={strided}", case.id);
                        assert_eq!(reference.dry, sources[row].dry[side], "{label}");
                        assert_eq!(
                            reference.physical_wet_eligible,
                            reference.physical_eligible && !reference.dry,
                            "{label}"
                        );
                        assert!(!(reference.rectangular && reference.pointed), "{label}");
                        assert_eq!(
                            reference.physical_eligible,
                            reference.within_first_band
                                && (reference.rectangular || reference.pointed),
                            "{label}"
                        );
                        let eligible = actual.eligible[row];
                        assert!(eligible == 0.0 || eligible == 1.0, "{label}");
                        if reference.film_exact {
                            assert_eq!(
                                eligible, 1.0,
                                "{label}: exact film source requires analytic eligibility"
                            );
                        }
                        if eligible == 0.0 {
                            zero(actual.potential[row], &label);
                            zero(actual.sound[row], &label);
                            continue;
                        }
                        assert!(
                            reference.physical_eligible,
                            "{label}: analytic source exceeds physical eligibility"
                        );
                        check(
                            phi_bounds[row],
                            reference.phi,
                            actual.potential[row],
                            &format!("{label}, potential"),
                        );
                        check(
                            sound_bounds[row],
                            reference.sound,
                            actual.sound[row],
                            &format!("{label}, sound"),
                        );
                        if reference.film_exact {
                            assert_eq!(
                                actual.potential[row].map(f64::from_bits),
                                [2.0, 0.0, 0.0],
                                "{label}"
                            );
                            assert_eq!(
                                actual.sound[row].map(f64::from_bits),
                                [1.0, 0.0, 0.0],
                                "{label}"
                            );
                        }
                    }
                    for component in [
                        value.potential.high(),
                        value.potential.low(),
                        value.potential.abs_error(),
                        value.sound.high(),
                        value.sound.low(),
                        value.sound.abs_error(),
                    ] {
                        assert_eq!(component.shape(), &[sources.len()]);
                        assert_eq!(component.dtype(), DType::F64);
                        assert!(component.device().is_same(client.device()));
                        for input in [&captured.left_area, &captured.right_area] {
                            for source in [input.high(), input.low(), input.abs_error()] {
                                assert!(!component.shares_storage_with(source));
                            }
                        }
                        for source in [
                            left_geometry.depth_m().lower(),
                            left_geometry.depth_m().upper(),
                            right_geometry.depth_m().lower(),
                            right_geometry.depth_m().upper(),
                        ] {
                            assert!(!component.shares_storage_with(source));
                        }
                    }
                }
                if let Some(expected) = &contiguous {
                    assert_eq!(&snapshots, expected);
                }
                let retained = calculate()?;
                for value in &result {
                    for tensor in [
                        &value.eligible,
                        value.potential.high(),
                        value.potential.low(),
                        value.potential.abs_error(),
                        value.sound.high(),
                        value.sound.low(),
                        value.sound.abs_error(),
                    ] {
                        client.copy_into(
                            tensor,
                            &Tensor::zeros(tensor.shape(), DType::F64, client.device())?,
                        )?;
                    }
                }
                assert_eq!(
                    retained.iter().map(snapshot).collect::<Result<Vec<_>>>()?,
                    snapshots
                );
                assert_eq!(
                    calculate()?
                        .iter()
                        .map(snapshot)
                        .collect::<Result<Vec<_>>>()?,
                    snapshots
                );
                assert_eq!(section.definitions(), definitions.as_slice());
                contiguous = Some(snapshots);
            }
            exercised += cases.len();
        }
        assert_eq!(exercised, CASES.len());
        ambiguous_support::<R>(&client)
    }

    fn ambiguous_support<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = batch::<R>(client)?;
        let geometry = HydraulicGeometry::from_batch(&section);
        for rectangular in [true, false] {
            let case = CASES
                .iter()
                .find(|case| {
                    case.sides[0].physical_wet_eligible
                        && if rectangular {
                            case.sides[0].rectangular
                        } else {
                            case.sides[0].pointed
                        }
                })
                .ok_or("Missing ordinary eligible analytic source")?;
            let source = support::cases()
                .get(case.source_index)
                .ok_or("Missing frozen analytic source")?;
            let owners =
                Tensor::from_slice(&[i64::try_from(source.owner)?], &[1], client.device())?;
            let area = expansion::<R>(client, &[source.inputs[0]], false)?;
            let recovered = section.recover_area_points_tensor(
                &owners,
                &area,
                &PrismaticRecoveryConfig::default(),
            )?;
            let (certain, uncertain) = hydraulic(error::checked::<R, _>(
                client,
                "conservative analytic source support",
                |a| {
                    let mut bands = geometry.gather(a, &owners)?;
                    let certain = analytic_bottom::source(
                        a,
                        &bands,
                        &area,
                        recovered.depth_m(),
                        f64::from_bits(case.gravity_bits),
                    )?;
                    // Widen the certificate without modifying the original profile or its physical source.
                    bands.end_area = a.bounds(
                        &Tensor::zeros(
                            bands.end_area.lower().shape(),
                            DType::F64,
                            client.device(),
                        )?,
                        bands.end_area.upper(),
                    )?;
                    let uncertain = analytic_bottom::source(
                        a,
                        &bands,
                        &area,
                        recovered.depth_m(),
                        f64::from_bits(case.gravity_bits),
                    )?;
                    Ok((certain, uncertain))
                },
            ))?;
            assert_eq!(certain.eligible.try_to_vec::<f64>()?, [1.0]);
            assert_eq!(uncertain.eligible.try_to_vec::<f64>()?, [0.0]);
            for (value, reference, label) in [
                (
                    &certain.potential,
                    case.sides[0].phi,
                    "ordinary analytic potential",
                ),
                (
                    &certain.sound,
                    case.sides[0].sound,
                    "ordinary analytic sound",
                ),
            ] {
                check(
                    endpoints(&bounds::<R>(client, value)?)?[0],
                    reference,
                    components(value)?[0],
                    label,
                );
            }
            for value in [&uncertain.potential, &uncertain.sound] {
                zero(components(value)?[0], "unavailable ambiguous source");
            }
        }
        Ok(())
    }

    #[test]
    fn configured_cpu_analytic_sources_retain_independent_potential_and_sound_certificates()
    -> Result {
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
    fn cuda_analytic_sources_retain_independent_potential_and_sound_certificates() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
