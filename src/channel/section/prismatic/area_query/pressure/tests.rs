use super::super::tests::{fixtures as queries, test_support as support};
use super::{pressure_range, types::PressureEndpoint};
use crate::channel::{
    flow::surveyed::error,
    section::{HydraulicGeometry, PrismaticRecoveryConfig},
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::collections::BTreeMap;
use support::{Client, Result, hydraulic};

#[path = "../pressure_fixtures/mod.rs"]
pub(in crate::channel::section::prismatic::area_query) mod fixtures;
use fixtures::{CASES, Case, DeltaAreaAuthority, Endpoint, QuantityPolicy, QuantityRef};

#[derive(Debug, PartialEq)]
struct Snapshot {
    area: Vec<[f64; 2]>,
    increment: Vec<[f64; 2]>,
    secant: Vec<[f64; 2]>,
    retained: Vec<[u64; 3]>,
}

fn interval<R: Runtime<DType = DType>>(value: &Interval<R>) -> Result<Vec<[f64; 2]>> {
    Ok(value
        .lower()
        .try_to_vec::<f64>()?
        .into_iter()
        .zip(value.upper().try_to_vec::<f64>()?)
        .map(|(lo, hi)| [lo, hi])
        .collect())
}

fn snapshot<R>(client: &R::Client, value: &PressureEndpoint<R>) -> Result<Snapshot>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let area = hydraulic(error::checked::<R, _>(
        client,
        "pressure area projection",
        |a| Ok(a.expansion_interval(&value.delta_area)?),
    ))?;
    let high = value.retained_increment.high().try_to_vec::<f64>()?;
    let low = value.retained_increment.low().try_to_vec::<f64>()?;
    let radius = value.retained_increment.abs_error().try_to_vec::<f64>()?;
    Ok(Snapshot {
        area: interval(&area)?,
        increment: interval(&value.physical_increment)?,
        secant: interval(&value.secant)?,
        retained: high
            .into_iter()
            .zip(low)
            .zip(radius)
            .map(|((hi, lo), radius)| [hi.to_bits(), lo.to_bits(), radius.to_bits()])
            .collect(),
    })
}

fn quantity(
    actual: [f64; 2],
    reference: QuantityRef,
    retained: Option<[u64; 3]>,
    label: &str,
) -> Result {
    let expected = reference.bounds.map(f64::from_bits);
    assert!(
        actual[0].is_finite() && actual[1].is_finite() && actual[0] <= actual[1],
        "{label}: {actual:?}"
    );
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
    match reference.policy {
        QuantityPolicy::ExactZero => {
            assert_eq!(actual, [0.0; 2], "{label}");
            assert_eq!(reference.native_allowance, Some(0));
            if let Some(raw) = retained {
                assert_eq!(raw.map(f64::from_bits), [0.0; 3], "{label}");
            }
        }
        QuantityPolicy::Local | QuantityPolicy::DecimalGeometry => {
            let allowance = reference
                .native_allowance
                .map(f64::from_bits)
                .ok_or("Frozen represented quantity needs an allowance")?;
            let error = (actual[0] - expected[1])
                .abs()
                .max((actual[1] - expected[0]).abs());
            assert!(
                error <= allowance,
                "{label}: error={error}, allowance={allowance}"
            );
        }
        QuantityPolicy::Underflow {
            physical_projected_bounds,
            maximum_physical_width,
            nonzero_physical,
        } => {
            let physical = physical_projected_bounds.map(f64::from_bits);
            assert!(
                actual[0] >= physical[0] && actual[1] <= physical[1],
                "{label}: {actual:?} exceeds {physical:?}"
            );
            assert!(
                actual[1] - actual[0] <= f64::from_bits(maximum_physical_width),
                "{label}"
            );
            assert!(nonzero_physical);
            assert_eq!(reference.native_allowance, None);
            if let Some(raw) = retained {
                let raw = raw.map(f64::from_bits);
                if raw[0] == 0.0 && raw[1] == 0.0 {
                    assert!(
                        raw[2] > 0.0,
                        "{label}: nonzero pressure lost its certificate"
                    );
                }
            }
        }
    }
    Ok(())
}

fn check(actual: &Snapshot, row: usize, reference: Endpoint, label: &str) -> Result {
    let expected = reference.delta_area.map(f64::from_bits);
    let area = actual.area[row];
    assert!(
        area[0] <= expected[0] && expected[1] <= area[1],
        "{label}: delta area"
    );
    let error = (area[0] - expected[1])
        .abs()
        .max((area[1] - expected[0]).abs());
    assert!(
        error <= f64::from_bits(reference.delta_area_allowance),
        "{label}: delta-area error={error}"
    );
    if reference.structural_equality {
        assert_eq!(
            reference.delta_area_authority,
            DeltaAreaAuthority::ExactZero
        );
        assert_eq!(area, [0.0; 2]);
    }
    quantity(
        actual.increment[row],
        reference.delta_pressure,
        Some(actual.retained[row]),
        &format!("{label}, pressure"),
    )?;
    quantity(
        actual.secant[row],
        reference.secant,
        None,
        &format!("{label}, secant"),
    )
}

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(CASES.len(), 22);
    let section = support::batch::<R>(&client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let table = hydraulic(error::checked::<R, _>(
        &client,
        "pressure model boundaries",
        |a| geometry.boundary_areas(a),
    ))?;
    let mut groups: BTreeMap<_, Vec<&Case>> = BTreeMap::new();
    for case in CASES.iter().rev() {
        let chart = queries::CASES
            .get(case.query_index)
            .ok_or("Missing pressure query")?;
        assert_eq!(chart.id, case.query_id);
        groups
            .entry((chart.chart.kind, chart.chart.add))
            .or_default()
            .push(case);
    }
    let mut count = 0;
    for cases in groups.values() {
        let charts: Vec<_> = cases
            .iter()
            .map(|case| queries::CASES[case.query_index].chart)
            .collect();
        let sources: Vec<_> = cases
            .iter()
            .map(|case| [case.source_components[0], case.source_components[1], 0])
            .collect();
        let mut baseline = None;
        for strided in [false, true] {
            let input = support::input::<R>(&client, &section, &charts, strided, false)?;
            let areas = support::expansion::<R>(&client, &sources, strided)?;
            let source = section.recover_area_points_tensor(
                &input.owners,
                &areas,
                &PrismaticRecoveryConfig::default(),
            )?;
            let calculate = || {
                hydraulic(error::checked::<R, _>(
                    &client,
                    "independent query pressure",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let supported =
                            support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                        let boundaries = table.gather(a, &input.owners)?;
                        pressure_range(a, &geometry, &bands, &supported, &source, &boundaries, 1.0)
                    },
                ))
            };
            let value = calculate()?;
            let state = [
                snapshot::<R>(&client, &value.lower)?,
                snapshot::<R>(&client, &value.upper)?,
            ];
            for (row, case) in cases.iter().enumerate() {
                check(
                    &state[0],
                    row,
                    case.endpoints[case.minimum_endpoint],
                    &format!("{}, lower", case.id),
                )?;
                check(
                    &state[1],
                    row,
                    case.endpoints[case.maximum_endpoint],
                    &format!("{}, upper", case.id),
                )?;
            }
            if let Some(expected) = &baseline {
                assert_eq!(&state, expected);
            }
            client.copy_into(
                value.lower.retained_increment.high(),
                &Tensor::zeros(
                    value.lower.retained_increment.high().shape(),
                    DType::F64,
                    client.device(),
                )?,
            )?;
            let fresh = calculate()?;
            assert_eq!(
                [
                    snapshot::<R>(&client, &fresh.lower)?,
                    snapshot::<R>(&client, &fresh.upper)?
                ],
                state
            );
            baseline = Some(state);
        }
        count += cases.len();
    }
    assert_eq!(count, CASES.len());
    bounded_weights::<R>(&client)
}

fn bounded_weights<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let minimum = f64::from_bits(1);
    let numerator_lower =
        Tensor::from_slice(&[0.0, minimum, 4.0 * minimum, 1e-14], &[4], client.device())?;
    let numerator_upper = Tensor::from_slice(
        &[1e-14, minimum, 5.0 * minimum, 2e-14],
        &[4],
        client.device(),
    )?;
    let denominator_lower = Tensor::from_slice(
        &[minimum, 2.0 * minimum, 4.0 * minimum, 1e-14],
        &[4],
        client.device(),
    )?;
    let denominator_upper = Tensor::from_slice(
        &[minimum, 2.0 * minimum, 8.0 * minimum, 1e-14],
        &[4],
        client.device(),
    )?;
    let result = hydraulic(error::checked::<R, _>(
        client,
        "bounded physical band weights",
        |a| {
            let numerator = a.bounds(&numerator_lower, &numerator_upper)?;
            let denominator = a.bounds(&denominator_lower, &denominator_upper)?;
            let zero = crate::channel::flow::surveyed::equation_ops::scalar(a, 0.0)?;
            let one = crate::channel::flow::surveyed::equation_ops::scalar(a, 1.0)?;
            super::weights::band_weights(a, &numerator, &denominator, &zero, &one)
        },
    ))?;
    let actual = interval(&result)?;
    let expected = [[0.0, 1.0], [0.5, 0.5], [0.5, 1.0], [1.0, 1.0]];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(0.0 <= actual[0] && actual[1] <= 1.0);
        assert!(actual[0] <= expected[0] && expected[1] <= actual[1]);
        let excess = (expected[0] - actual[0]).max(actual[1] - expected[1]);
        assert!(excess <= 4.0 * f64::EPSILON);
    }
    assert_eq!(actual[0], expected[0]);
    assert_eq!(actual[3], expected[3]);
    Ok(())
}

#[test]
fn cpu_pressure_paths_retain_local_increments_and_direct_secants() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let previous = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    matrix::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), previous);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_pressure_paths_retain_local_increments_and_direct_secants() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
