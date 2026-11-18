use super::types::CapturedPointInput;
use crate::channel::flow::surveyed::equation_ops::choose_expansion;
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

#[path = "source_velocity_quotient.rs"]
mod quotient_terms;
use quotient_terms::{conditioned_difference, identical, masked, quotient, same_discharge_sign};
#[path = "source_velocity_selection.rs"]
mod selection;

/// Physical quotient certificates bound to their captured conserved inputs.
pub(crate) struct SourceVelocity<R: Runtime<DType = DType>> {
    input: Arc<CapturedPointInput<R>>,
    pub(crate) left: CertifiedExpansion<R>,
    pub(crate) right: CertifiedExpansion<R>,
    pub(crate) difference: CertifiedExpansion<R>,
    /// One only where both physical source velocities are defined.
    pub(crate) both_wet: Tensor<R>,
    /// Wet rows with exactly identical normalized area and discharge points.
    pub(crate) uniform: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SourceVelocity<R> {
    pub(crate) fn input(&self) -> &CapturedPointInput<R> {
        &self.input
    }
}

pub(crate) fn prepare<R>(
    client: &R::Client,
    input: &Arc<CapturedPointInput<R>>,
) -> HydraulicResult<SourceVelocity<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    checked::<R, _>(client, "surveyed point source velocities", |a| {
        let zero = Tensor::zeros(input.owners.shape(), a.dtype(), client.device())?;
        let one = Tensor::ones(input.owners.shape(), a.dtype(), client.device())?;
        let left_wet = client.eq(&input.left_dry, &zero)?;
        let right_wet = client.eq(&input.right_dry, &zero)?;
        let both_wet = client.mul(&left_wet, &right_wet)?;
        let left = quotient(
            a,
            &left_wet,
            &input.left_discharge,
            &input.left_area,
            (&zero, &one),
        )?;
        let right = quotient(
            a,
            &right_wet,
            &input.right_discharge,
            &input.right_area,
            (&zero, &one),
        )?;
        let equal_area = identical(client, &input.left_area, &input.right_area)?;
        let equal_discharge = identical(client, &input.left_discharge, &input.right_discharge)?;
        let uniform = client.mul(&both_wet, &client.mul(&equal_area, &equal_discharge)?)?;

        let same_sign = same_discharge_sign(client, input, &zero)?;
        let eligible = client.mul(
            &both_wet,
            &client.mul(&same_sign, &client.eq(&uniform, &zero)?)?,
        )?;
        let left_area = masked(a, &eligible, &input.left_area, &one, &zero)?;
        let right_area = masked(a, &eligible, &input.right_area, &one, &zero)?;
        let delta_area = a.expansion_sub(&left_area, &right_area)?;
        let difference_bounds = a.expansion_interval(&delta_area)?;
        let left_order = client.ge(difference_bounds.lower(), &zero)?;
        let right_order = client.le(difference_bounds.upper(), &zero)?;

        // These differences prove comparability without first doubling a possibly large area.
        let left_change = masked(a, &left_order, &delta_area, &zero, &zero)?;
        let left_other = masked(a, &left_order, &right_area, &one, &zero)?;
        let left_margin = a.expansion_sub(&left_other, &left_change)?;
        let left_margin = a.expansion_interval(&left_margin)?;
        let left_mask = client.mul(
            &eligible,
            &client.mul(&left_order, &client.ge(left_margin.lower(), &zero)?)?,
        )?;
        let right_change = masked(a, &right_order, &delta_area, &zero, &zero)?;
        let right_other = masked(a, &right_order, &left_area, &one, &zero)?;
        let right_margin = a.expansion_add(&right_other, &right_change)?;
        let right_margin = a.expansion_interval(&right_margin)?;
        let right_mask = client.mul(
            &eligible,
            &client.mul(
                &client.eq(&left_mask, &zero)?,
                &client.mul(&right_order, &client.ge(right_margin.lower(), &zero)?)?,
            )?,
        )?;
        let conditioned = client.maximum(&left_mask, &right_mask)?;
        let fallback = client.mul(
            &both_wet,
            &client.mul(
                &client.eq(&uniform, &zero)?,
                &client.eq(&conditioned, &zero)?,
            )?,
        )?;

        // Only certified conditioned rows enter the difference-first numerator.
        let left_discharge = masked(a, &conditioned, &input.left_discharge, &zero, &zero)?;
        let right_discharge = masked(a, &conditioned, &input.right_discharge, &zero, &zero)?;
        let delta_discharge = a.expansion_sub(&right_discharge, &left_discharge)?;
        let left_difference = conditioned_difference(
            a,
            &left_mask,
            &delta_discharge,
            &delta_area,
            &left,
            &input.right_area,
            (&zero, &one),
        )?;
        let right_difference = conditioned_difference(
            a,
            &right_mask,
            &delta_discharge,
            &delta_area,
            &right,
            &input.left_area,
            (&zero, &one),
        )?;
        let fallback_left = masked(a, &fallback, &left, &zero, &zero)?;
        let fallback_right = masked(a, &fallback, &right, &zero, &zero)?;
        let independent = a.expansion_sub(&fallback_right, &fallback_left)?;
        let difference = choose_expansion(a, &right_mask, &right_difference, &independent)?;
        let difference = choose_expansion(a, &left_mask, &left_difference, &difference)?;
        Ok(SourceVelocity {
            input: Arc::clone(input),
            left,
            right,
            difference,
            both_wet,
            uniform,
        })
    })
}

#[cfg(test)]
mod tests {
    use super::super::source_fixture_support::{
        Client, Result, batch, bounds, components, contained, endpoints, expansion, hydraulic, zero,
    };
    use super::super::source_fixtures::{CASES, Case, PROFILES};
    use super::{self as source_velocity};
    use crate::channel::flow::surveyed::error;
    use numr::dtype::DType;
    use numr::ops::BinaryOps;
    use numr::runtime::{Runtime, RuntimeClient};
    use numr::tensor::Tensor;
    use std::collections::BTreeSet;

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
            let captured =
                hydraulic(section.capture_riemann_points(
                    &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3],
                ))?;
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

    mod admission {
        use super::{Client, PROFILES, Result, batch, expansion};
        use crate::channel::flow::surveyed::error::{HydraulicError, HydraulicResult};
        use crate::channel::section::CrossSectionError;
        use numr::dtype::DType;
        use numr::runtime::{Runtime, RuntimeClient};
        use numr::tensor::Tensor;

        fn invalid<T>(result: HydraulicResult<T>) -> Result {
            let error = result
                .err()
                .ok_or("Source capture accepts an invalid selected point")?;
            assert!(
                matches!(
                    error,
                    HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
                ),
                "{error:?}"
            );
            Ok(())
        }

        pub(super) fn run<R>(client: R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R>,
        {
            let section = batch::<R>(&client)?;
            let owner = Tensor::<R>::from_slice(&[0i64], &[1], client.device())?;
            let one = 1.0f64.to_bits();
            let baseline = [[one, 0, 0], [0; 3], [one, 0, 0], [0; 3]];
            let make = |points: [[u64; 3]; 4]| {
                points
                    .iter()
                    .map(|point| expansion::<R>(&client, &[*point], false))
                    .collect::<Result<Vec<_>>>()
            };
            let input = make(baseline)?;
            for owner_value in [-1, i64::try_from(PROFILES.len())?] {
                let invalid_owner = Tensor::<R>::from_slice(&[owner_value], &[1], client.device())?;
                invalid(section.capture_riemann_points(
                    &invalid_owner,
                    &input[0],
                    &input[1],
                    &input[2],
                    &input[3],
                ))?;
            }
            let scalar_owner = Tensor::<R>::from_slice(&[0i64], &[], client.device())?;
            let float_owner = Tensor::<R>::from_slice(&[0.0f64], &[1], client.device())?;
            for invalid_owner in [&scalar_owner, &float_owner] {
                invalid(section.capture_riemann_points(
                    invalid_owner,
                    &input[0],
                    &input[1],
                    &input[2],
                    &input[3],
                ))?;
            }
            let extra = expansion::<R>(&client, &[[one, 0, 0]; 2], false)?;
            invalid(
                section.capture_riemann_points(&owner, &extra, &input[1], &input[2], &input[3]),
            )?;
            for field in 0..4 {
                let mut points = baseline;
                points[field][2] = 1;
                let input = make(points)?;
                invalid(
                    section
                        .capture_riemann_points(&owner, &input[0], &input[1], &input[2], &input[3]),
                )?;
            }
            for area_field in [0, 2] {
                let mut points = baseline;
                points[area_field] = [0, (-f64::from_bits(1)).to_bits(), 0];
                let input = make(points)?;
                invalid(
                    section
                        .capture_riemann_points(&owner, &input[0], &input[1], &input[2], &input[3]),
                )?;
                for discharge in [f64::from_bits(1), -f64::from_bits(1)] {
                    let mut points = baseline;
                    points[area_field] = [0; 3];
                    points[area_field + 1] = [0, discharge.to_bits(), 0];
                    let input = make(points)?;
                    invalid(section.capture_riemann_points(
                        &owner, &input[0], &input[1], &input[2], &input[3],
                    ))?;
                }
            }
            Ok(())
        }
    }
}
