use super::endpoints::PhysicalEndpoint;
use super::{fixtures, test_support as support};
use crate::channel::flow::surveyed::error;
use crate::channel::section::HydraulicGeometry;
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use std::collections::BTreeMap;
use support::{Client, Result, hydraulic};

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(&client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let mut groups = BTreeMap::new();
    for case in fixtures::CASES.iter().rev() {
        groups
            .entry((case.chart.kind, case.chart.add))
            .or_insert_with(Vec::new)
            .push(case);
    }
    let mut count = 0;
    for cases in groups.values() {
        let charts: Vec<_> = cases.iter().map(|case| case.chart).collect();
        let input = support::input::<R>(&client, &section, &charts, true, false)?;
        let last = i64::try_from(cases.len() - 1)?;
        let rows = Tensor::from_slice(&[last, 0, last], &[3], client.device())?;
        for (physical, upper) in [
            (PhysicalEndpoint::Lower, false),
            (PhysicalEndpoint::Upper, true),
        ] {
            let (endpoint, selected, value, parent_rows, selected_rows) = hydraulic(
                error::checked::<R, _>(&client, "physical area endpoints", |a| {
                    let bands = geometry.gather(a, &input.owners)?;
                    let parent = support::query(a, &bands, &input)?.prove_support(a, &bands)?;
                    let endpoint = parent.endpoint(a, &bands, physical)?;
                    let value = geometry.recover_query(a, &bands, &endpoint)?;
                    let selected_bands = bands.select(a, &rows)?;
                    let selected_parent = parent.select(a, &rows)?;
                    let selected = selected_parent.endpoint(a, &selected_bands, physical)?;
                    let parent_rows = selected_parent.query.original_rows.try_to_vec::<i64>()?;
                    let selected_rows = selected.query.original_rows.try_to_vec::<i64>()?;
                    Ok((endpoint, selected, value, parent_rows, selected_rows))
                }),
            )?;
            assert_eq!(parent_rows, [last, 0, last]);
            assert_eq!(selected_rows, parent_rows);
            let fields = [
                &value.state.area,
                &value.state.forward_area,
                &value.state.depth,
                &value.state.top_width,
                &value.state.integral,
                &value.state.hydraulic_depth,
            ];
            for (field, reference_index) in fields.into_iter().zip([0, 0, 1, 2, 3, 4]) {
                let lower = field.lower().try_to_vec::<f64>()?;
                let higher = field.upper().try_to_vec::<f64>()?;
                for (row, case) in cases.iter().enumerate() {
                    let reference = case.fields[reference_index];
                    let expected = if upper {
                        reference.maximum
                    } else {
                        reference.minimum
                    }
                    .map(f64::from_bits);
                    let actual = [lower[row], higher[row]];
                    assert!(
                        actual[0] <= expected[0] && expected[1] <= actual[1],
                        "{}, upper={upper}, field={reference_index}: {actual:?} excludes {expected:?}",
                        case.id
                    );
                    let error = (actual[0] - expected[1])
                        .abs()
                        .max((actual[1] - expected[0]).abs());
                    assert!(
                        error <= f64::from_bits(reference.allowances[usize::from(upper)]),
                        "{}, upper={upper}, field={reference_index}: error={error}",
                        case.id
                    );
                }
            }
            let selected_owners = selected.query.owners.try_to_vec::<i64>()?;
            client.copy_into(
                &endpoint.query.owners,
                &Tensor::zeros(endpoint.query.owners.shape(), DType::I64, client.device())?,
            )?;
            assert_eq!(selected.query.owners.try_to_vec::<i64>()?, selected_owners);
            let selected_parameters = selected.query.parameter.lower().try_to_vec::<f64>()?;
            client.copy_into(
                endpoint.query.parameter.lower(),
                &Tensor::zeros(
                    endpoint.query.parameter.lower().shape(),
                    DType::F64,
                    client.device(),
                )?,
            )?;
            assert_eq!(
                selected.query.parameter.lower().try_to_vec::<f64>()?,
                selected_parameters
            );
        }
        count += cases.len();
    }
    assert_eq!(count, 17);
    Ok(())
}

#[test]
fn cpu_physical_endpoints_preserve_map_geometry_and_original_rows() -> Result {
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
fn cuda_physical_endpoints_preserve_map_geometry_and_original_rows() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
