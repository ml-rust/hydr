use super::source_velocity;
use crate::channel::flow::surveyed::error;
use numr::dtype::DType;
use numr::ops::BinaryOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeSet;

#[path = "source_fixtures.rs"]
mod fixtures;
use fixtures::{CASES, Case};
#[path = "source_admission_tests.rs"]
mod admission;
#[path = "source_test_support.rs"]
pub(super) mod support;
use support::{
    Client, Result, batch, bounds, components, contained, endpoints, expansion, hydraulic, zero,
};

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    assert_eq!(CASES.len(), 70);
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.group)
            .collect::<BTreeSet<_>>()
            .len(),
        61
    );
    let section = batch::<R>(&client)?;
    let cases: Vec<&Case> = CASES.iter().rev().collect();
    let owner_values: Vec<_> = cases
        .iter()
        .map(|case| i64::try_from(case.owner))
        .collect::<std::result::Result<_, _>>()?;
    let mut baseline = None;
    for strided in [false, true] {
        let owners = if strided {
            let storage: Vec<_> = owner_values
                .iter()
                .flat_map(|&owner| [i64::MAX, owner])
                .collect();
            let view = Tensor::from_slice(&storage, &[cases.len(), 2], client.device())?
                .narrow(1, 1, 1)?
                .squeeze(Some(1));
            assert!(!view.is_contiguous());
            assert!(view.offset() > 0);
            view
        } else {
            Tensor::from_slice(&owner_values, &[cases.len()], client.device())?
        };
        let inputs = (0..4)
            .map(|field| {
                expansion::<R>(
                    &client,
                    &cases
                        .iter()
                        .map(|case| case.inputs[field])
                        .collect::<Vec<_>>(),
                    strided,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let expected = inputs.iter().map(components).collect::<Result<Vec<_>>>()?;
        let captured = hydraulic(
            section.capture_riemann_points(&owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3]),
        )?;
        let captured_values = [
            &captured.left_area,
            &captured.left_discharge,
            &captured.right_area,
            &captured.right_discharge,
        ];
        assert_eq!(captured.owners.try_to_vec::<i64>()?, owner_values);
        for (value, expected) in captured_values.into_iter().zip(&expected) {
            assert_eq!(components(value)?, *expected);
        }
        for (side, dry) in [&captured.left_dry, &captured.right_dry]
            .into_iter()
            .enumerate()
        {
            assert_eq!(
                dry.try_to_vec::<f64>()?,
                cases
                    .iter()
                    .map(|case| if case.dry[side] { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>()
            );
        }
        let velocity = hydraulic(source_velocity::prepare::<R>(&client, &captured))?;
        let output_values = [&velocity.left, &velocity.right, &velocity.difference];
        let output_components = output_values
            .into_iter()
            .map(components)
            .collect::<Result<Vec<_>>>()?;
        let projected = output_values
            .into_iter()
            .map(|value| endpoints(&bounds::<R>(&client, value)?))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            velocity.both_wet.try_to_vec::<f64>()?,
            cases
                .iter()
                .map(|case| if case.dry.iter().all(|&dry| !dry) {
                    1.0
                } else {
                    0.0
                })
                .collect::<Vec<_>>()
        );
        assert_eq!(
            velocity.uniform.try_to_vec::<f64>()?,
            cases
                .iter()
                .map(|case| if case.uniform { 1.0 } else { 0.0 })
                .collect::<Vec<_>>()
        );
        let reference_points: Vec<_> = cases
            .iter()
            .map(|case| {
                case.difference.map_or([0; 3], |reference| {
                    [reference.center[0], reference.center[1], reference.residual]
                })
            })
            .collect();
        let reference = expansion::<R>(&client, &reference_points, false)?;
        let error = hydraulic(error::checked::<R, _>(
            &client,
            "source difference reference distance",
            |a| {
                let error = a.expansion_sub(&velocity.difference, &reference)?;
                Ok(a.expansion_interval(&error)?)
            },
        ))?;
        let error = endpoints(&error)?;
        for (index, case) in cases.iter().enumerate() {
            for field in 0..3 {
                let reference = if field < 2 {
                    case.velocity[field]
                } else {
                    case.difference
                };
                let label = format!("{}, field={field}, strided={strided}", case.id);
                let point = output_components[field][index];
                let radius = f64::from_bits(point[2]);
                assert!(radius.is_finite() && radius >= 0.0, "{label}");
                if let Some(reference) = reference {
                    contained(projected[field][index], reference, &label);
                    if reference.non_dyadic {
                        assert!(radius > 0.0, "{label}: physical quotient arithmetic radius");
                    }
                } else {
                    zero(point, &label);
                }
            }
            if case.uniform {
                zero(output_components[2][index], case.id);
            }
            if case.difference_zero {
                assert!(
                    projected[2][index][0] <= 0.0 && projected[2][index][1] >= 0.0,
                    "{}",
                    case.id
                );
            }
            if let Some(allowance) = case.difference_allowance {
                let allowance = f64::from_bits(allowance);
                let maximum = error[index][0].abs().max(error[index][1].abs());
                assert!(
                    maximum <= allowance,
                    "{}: difference error={maximum}, frozen allowance={allowance}",
                    case.id
                );
            }
        }
        if let Some(expected) = &baseline {
            assert_eq!(&output_components, expected);
        }
        for tensor in inputs
            .iter()
            .flat_map(|value| [value.high(), value.low(), value.abs_error()])
            .chain(std::iter::once(&owners))
        {
            client.copy_into(
                tensor,
                &Tensor::zeros(tensor.shape(), tensor.dtype(), client.device())?,
            )?;
        }
        assert_eq!(captured.owners.try_to_vec::<i64>()?, owner_values);
        for (value, expected) in captured_values.into_iter().zip(&expected) {
            assert_eq!(components(value)?, *expected);
        }
        let repeated = hydraulic(source_velocity::prepare::<R>(&client, &captured))?;
        for tensor in output_values
            .into_iter()
            .flat_map(|value| [value.high(), value.low(), value.abs_error()])
        {
            client.copy_into(
                tensor,
                &Tensor::zeros(tensor.shape(), DType::F64, client.device())?,
            )?;
        }
        assert_eq!(
            [&repeated.left, &repeated.right, &repeated.difference]
                .into_iter()
                .map(components)
                .collect::<Result<Vec<_>>>()?,
            output_components
        );
        let fresh = hydraulic(source_velocity::prepare::<R>(&client, &captured))?;
        assert_eq!(
            [&fresh.left, &fresh.right, &fresh.difference]
                .into_iter()
                .map(components)
                .collect::<Result<Vec<_>>>()?,
            output_components
        );
        baseline = Some(output_components);
    }
    Ok(())
}

#[test]
fn configured_cpu_source_points_preserve_quotients_and_relative_velocity() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let previous = original.parallelism();
    let config = ParallelismConfig::new(Some(2), Some(257));
    let client = original.with_parallelism(config);
    let retained = client.clone();
    matrix::<CpuRuntime>(client.clone())?;
    admission::run::<CpuRuntime>(client)?;
    assert_eq!(retained.parallelism(), config);
    assert_eq!(original.parallelism(), previous);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_source_points_preserve_quotients_and_relative_velocity() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    matrix::<CudaRuntime>(client.clone())?;
    admission::run::<CudaRuntime>(client)
}
