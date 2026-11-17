use super::source_difference::source_difference;
use super::tests::{fixtures as queries, test_support as support};
use crate::channel::flow::surveyed::error;
use crate::channel::section::{HydraulicGeometry, PrismaticRecoveryConfig};
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeMap;
use support::{Client, Result, hydraulic};

#[path = "source_difference_fixtures/mod.rs"]
mod fixtures;
use fixtures::{CASES, Case, DeltaAreaAuthority};

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
        "source boundary table",
        |a| geometry.boundary_areas(a),
    ))?;
    let mut groups = BTreeMap::new();
    for case in CASES.iter().rev() {
        let chart = queries::CASES
            .get(case.query_index)
            .ok_or("Missing query index")?;
        assert_eq!(chart.id, case.query_id);
        groups
            .entry((chart.chart.kind, chart.chart.add))
            .or_insert_with(Vec::new)
            .push(case);
    }
    let mut count = 0;
    for cases in groups.values() {
        let charts: Vec<_> = cases
            .iter()
            .map(|case| queries::CASES[case.query_index].chart)
            .collect();
        let points: Vec<_> = cases
            .iter()
            .map(|case| [case.source_components[0], case.source_components[1], 0])
            .collect();
        let mut baseline = None;
        for strided in [false, true] {
            let input = support::input::<R>(&client, &section, &charts, strided, false)?;
            let area = support::expansion::<R>(&client, &points, strided)?;
            let source = section.recover_area_points_tensor(
                &input.owners,
                &area,
                &PrismaticRecoveryConfig::default(),
            )?;
            let calculate = || {
                hydraulic(error::checked::<R, _>(
                    &client,
                    "retained source area differences",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let query = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                        let boundaries = table.gather(a, &input.owners)?;
                        source_difference(a, &bands, &query, &source, &boundaries)
                    },
                ))
            };
            let value = calculate()?;
            let actual = hydraulic(error::checked::<R, _>(
                &client,
                "source difference projection",
                |a| {
                    Ok([
                        a.expansion_interval(&value.lower)?,
                        a.expansion_interval(&value.upper)?,
                    ])
                },
            ))?;
            let mut snapshot = Vec::new();
            for (endpoint, interval) in actual.iter().enumerate() {
                let lower = interval.lower().try_to_vec::<f64>()?;
                let upper = interval.upper().try_to_vec::<f64>()?;
                let expansion = if endpoint == 0 {
                    &value.lower
                } else {
                    &value.upper
                };
                let high = expansion.high().try_to_vec::<f64>()?;
                let low = expansion.low().try_to_vec::<f64>()?;
                let radius = expansion.abs_error().try_to_vec::<f64>()?;
                for (row, case) in cases.iter().enumerate() {
                    check(
                        case,
                        endpoint,
                        [lower[row], upper[row]],
                        [high[row], low[row], radius[row]],
                    );
                    snapshot.push([
                        high[row].to_bits(),
                        low[row].to_bits(),
                        radius[row].to_bits(),
                    ]);
                }
                for component in [expansion.high(), expansion.low(), expansion.abs_error()] {
                    for original in [area.high(), area.low(), area.abs_error()] {
                        assert!(!component.shares_storage_with(original));
                    }
                }
            }
            if let Some(expected) = &baseline {
                assert_eq!(&snapshot, expected);
            }
            let replacement =
                Tensor::zeros(value.lower.high().shape(), DType::F64, client.device())?;
            client.copy_into(value.lower.high(), &replacement)?;
            let fresh = calculate()?;
            let fresh_high = fresh.lower.high().try_to_vec::<f64>()?;
            for (row, high) in fresh_high.iter().enumerate() {
                assert_eq!(high.to_bits(), snapshot[row][0]);
            }
            baseline = Some(snapshot);
        }
        count += cases.len();
    }
    assert_eq!(count, CASES.len());
    retained_uncertainty::<R>(&client)
}

fn retained_uncertainty<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let reference = CASES
        .iter()
        .find(|case| case.query_id == "first_band_start_signed_offsets")
        .ok_or("Missing original band-start source")?;
    let mut chart = queries::CASES[reference.query_index].chart;
    chart.parameter = [0; 2];
    let section = support::batch::<R>(client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let input = support::input::<R>(client, &section, &[chart], false, false)?;
    let area = support::expansion::<R>(
        client,
        &[[
            reference.source_components[0],
            reference.source_components[1],
            0,
        ]],
        false,
    )?;
    let source = section.recover_area_points_tensor(
        &input.owners,
        &area,
        &PrismaticRecoveryConfig::default(),
    )?;
    let results = hydraulic(error::checked::<R, _>(
        client,
        "retained boundary uncertainty",
        |a| {
            let bands = geometry.gather(a, &input.owners)?;
            let query = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
            let boundaries = geometry.boundary_areas(a)?.gather(a, &input.owners)?;
            let certain = source_difference(a, &bands, &query, &source, &boundaries)?;
            let radius = boundaries.areas().abs_error();
            client.copy_into(
                radius,
                &Tensor::full_scalar(radius.shape(), DType::F64, 1e-12, client.device())?,
            )?;
            let uncertain = source_difference(a, &bands, &query, &source, &boundaries)?;
            Ok([certain, uncertain])
        },
    ))?;
    for value in [&results[0].lower, &results[0].upper] {
        assert_eq!(value.high().try_to_vec::<f64>()?, [0.0]);
        assert_eq!(value.low().try_to_vec::<f64>()?, [0.0]);
        assert_eq!(value.abs_error().try_to_vec::<f64>()?, [0.0]);
    }
    for value in [&results[1].lower, &results[1].upper] {
        assert_eq!(value.high().try_to_vec::<f64>()?, [0.0]);
        assert_eq!(value.low().try_to_vec::<f64>()?, [0.0]);
        assert!(value.abs_error().try_to_vec::<f64>()?[0] > 0.0);
    }
    Ok(())
}

fn check(case: &Case, endpoint: usize, actual: [f64; 2], center: [f64; 3]) {
    let original = if endpoint == 0 {
        case.minimum_endpoint
    } else {
        case.maximum_endpoint
    };
    let reference = case.endpoints[original];
    let expected = reference.delta_area.map(f64::from_bits);
    let label = format!("{}, physical endpoint={endpoint}", case.id);
    assert!(
        actual[0] <= expected[0] && expected[1] <= actual[1],
        "{label}: {actual:?} excludes {expected:?}"
    );
    let error = (actual[0] - expected[1])
        .abs()
        .max((actual[1] - expected[0]).abs());
    assert!(
        error <= f64::from_bits(reference.delta_area_allowance),
        "{label}: error={error}"
    );
    if reference.structural_equality {
        assert_eq!(
            reference.delta_area_authority,
            DeltaAreaAuthority::ExactZero
        );
        assert_eq!(center, [0.0; 3], "{label}");
    }
}

#[test]
fn cpu_query_source_differences_retain_original_boundaries_and_offsets() -> Result {
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
fn cuda_query_source_differences_retain_original_boundaries_and_offsets() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
