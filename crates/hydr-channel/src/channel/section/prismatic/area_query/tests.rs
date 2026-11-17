use crate::channel::flow::surveyed::error;
use crate::channel::section::HydraulicGeometry;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::{BTreeMap, BTreeSet};

#[path = "admission_tests.rs"]
mod admission_tests;
#[path = "fixtures/mod.rs"]
pub(in crate::channel::section::prismatic) mod fixtures;
#[path = "parameter_rebinding_tests.rs"]
pub(super) mod parameter_rebinding_tests;
#[path = "test_support.rs"]
pub(in crate::channel::section::prismatic) mod test_support;
use fixtures::{AnchorKind, CASES, Case};
use test_support::{Client, Result, batch, check, hydraulic, input, query, snapshot};

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(CASES.len(), 17);
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.group)
            .collect::<BTreeSet<_>>()
            .len(),
        15
    );
    assert!(CASES.iter().any(|left| {
        CASES
            .iter()
            .any(|right| left.group == right.group && left.chart.add != right.chart.add)
    }));
    let section = batch::<R>(&client)?;
    let definitions = section.definitions().to_vec();
    let geometry = HydraulicGeometry::from_batch(&section);
    let mut groups: BTreeMap<(AnchorKind, bool), Vec<&Case>> = BTreeMap::new();
    for case in CASES.iter().rev() {
        groups
            .entry((case.chart.kind, case.chart.add))
            .or_default()
            .push(case);
    }
    let mut exercised = BTreeSet::new();
    for cases in groups.values() {
        let charts: Vec<_> = cases.iter().map(|case| case.chart).collect();
        let count = cases.len();
        let last = i64::try_from(count - 1)?;
        let row_values = [last, 0, last, 0];
        let row_indices = [count - 1, 0, count - 1, 0];
        let selected_cases: Vec<_> = row_indices.iter().map(|&index| cases[index]).collect();
        let expected_owners: Vec<_> = selected_cases
            .iter()
            .map(|case| i64::try_from(case.chart.owner))
            .collect::<std::result::Result<_, _>>()?;
        let rows = Tensor::<R>::from_slice(&row_values, &[row_values.len()], client.device())?;
        let mut contiguous = None;
        for strided in [false, true] {
            let input = input::<R>(&client, &section, &charts, strided, false)?;
            let (full, selected, selected_raw, selected_owners, original_rows, raw_original_rows) =
                hydraulic(error::checked::<R, _>(
                    &client,
                    "private area chart matrix",
                    |a| {
                        let bands = geometry.gather(a, &input.owners)?;
                        let query = query(a, &bands, &input)?;
                        let selected_bands = bands.select(a, &rows)?;
                        let selected_raw =
                            query.select(a, &rows)?.prove_support(a, &selected_bands)?;
                        let supported = query.prove_support(a, &bands)?;
                        let selected = supported.select(a, &rows)?;
                        let selected_owners = selected.query.owners.try_to_vec::<i64>()?;
                        let original_rows = selected.query.original_rows.try_to_vec::<i64>()?;
                        let raw_original_rows =
                            selected_raw.query.original_rows.try_to_vec::<i64>()?;
                        Ok((
                            geometry.recover_query(a, &bands, &supported)?,
                            geometry.recover_query(a, &selected_bands, &selected)?,
                            geometry.recover_query(a, &selected_bands, &selected_raw)?,
                            selected_owners,
                            original_rows,
                            raw_original_rows,
                        ))
                    },
                ))?;
            assert_eq!(selected_owners, expected_owners);
            assert_eq!(original_rows, row_values);
            assert_eq!(raw_original_rows, row_values);
            let expected = snapshot(&full)?;
            let retained_selected = snapshot(&selected)?;
            let retained_raw = snapshot(&selected_raw)?;
            check(&expected, cases);
            check(&retained_selected, &selected_cases);
            check(&retained_raw, &selected_cases);
            if let Some(contiguous) = &contiguous {
                assert_eq!(&expected, contiguous);
            }
            for interval in [
                &full.state.area,
                &full.state.forward_area,
                &full.state.depth,
                &full.state.top_width,
                &full.state.integral,
                &full.state.hydraulic_depth,
                full.coordinates.area_above_start_m2(),
                full.coordinates.area_below_end_m2(),
                full.coordinates.depth_above_start_m(),
                full.coordinates.depth_below_end_m(),
            ] {
                for endpoint in [interval.lower(), interval.upper()] {
                    client.copy_into(
                        endpoint,
                        &Tensor::zeros(endpoint.shape(), DType::F64, client.device())?,
                    )?;
                }
            }
            assert_eq!(snapshot(&selected)?, retained_selected);
            assert_eq!(snapshot(&selected_raw)?, retained_raw);
            let fresh = hydraulic(error::checked::<R, _>(
                &client,
                "independent private area chart",
                |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let supported = query(a, &bands, &input)?.prove_support(a, &bands)?;
                    geometry.recover_query(a, &bands, &supported)
                },
            ))?;
            assert_eq!(snapshot(&fresh)?, expected);
            assert_eq!(section.definitions(), definitions.as_slice());
            contiguous = Some(expected);
        }
        exercised.extend(cases.iter().map(|case| case.id));
    }
    assert_eq!(exercised.len(), 17);
    admission_tests::run::<R>(&client, &section)
}

#[test]
fn configured_cpu_private_area_charts_enclose_independent_extrema() -> Result {
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
fn cuda_private_area_charts_enclose_independent_extrema() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
