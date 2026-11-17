use super::super::{
    area_query::tests::{
        fixtures::CASES,
        test_support::{Client, Result, batch, hydraulic},
    },
    inputs::BankInputs,
};
use super::bank;
use crate::channel::{flow::surveyed::error, section::HydraulicGeometry};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

fn boundary_matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = batch::<R>(&client)?;
    let geometry = HydraulicGeometry::from_batch(&section);
    let owners = Tensor::<R>::from_slice(&[0_i64, 1, 2, 3, 4], &[5], client.device())?;
    hydraulic(error::checked::<R, _>(
        &client,
        "physical boundary area witnesses",
        |a| {
            let table = geometry.boundary_areas(a)?;
            let all = table.gather(a, &owners)?;
            assert_eq!(all.areas().high().shape(), &[5, 4]);
            assert_eq!(all.cap_indices().try_to_vec::<i64>()?, [2, 1, 3, 1, 1]);
            assert!(
                all.areas()
                    .abs_error()
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&r| r == 0.0)
            );

            let first = Tensor::from_slice(&[1_i64; 5], &[5], client.device())?;
            let high = Tensor::from_slice(
                &[f64::from_bits(0x3feccccccccccccd), 6.0, 3.0, 19.5, 13.5],
                &[5],
                client.device(),
            )?;
            let low = Tensor::from_slice(
                &[f64::from_bits(0xbc80000000000000), 0.0, 0.0, 0.0, 0.0],
                &[5],
                client.device(),
            )?;
            let zeros = Tensor::zeros(&[5], DType::F64, client.device())?;
            let source = a.expansion_bounds(&high, &low, &zeros)?;
            assert_eq!(
                all.exact_boundary_mask(a, &source, &first)?
                    .try_to_vec::<f64>()?,
                [1.0; 5]
            );
            let uncertain = a.expansion_bounds(
                source.high(),
                source.low(),
                &Tensor::full_scalar(&[5], DType::F64, f64::EPSILON, client.device())?,
            )?;
            assert_eq!(
                all.exact_boundary_mask(a, &uncertain, &first)?
                    .try_to_vec::<f64>()?,
                [0.0; 5]
            );
            let projected = a.expansion_point(&high)?;
            assert_eq!(
                all.exact_boundary_mask(a, &projected, &first)?
                    .try_to_vec::<f64>()?,
                [0.0, 1.0, 1.0, 1.0, 1.0]
            );

            let caps = all.boundary(a, all.cap_indices())?;
            let last = Tensor::from_slice(&[3_i64; 5], &[5], client.device())?;
            assert_eq!(
                all.exact_boundary_mask(a, &caps, &last)?
                    .try_to_vec::<f64>()?,
                [1.0; 5]
            );
            let middle_owner = Tensor::from_slice(&[2_i64, 2], &[2], client.device())?;
            let middle = table.gather(a, &middle_owner)?;
            let indices = Tensor::from_slice(&[2_i64, 3], &[2], client.device())?;
            let physical =
                a.expansion_point(&Tensor::from_slice(&[7.5, 14.0], &[2], client.device())?)?;
            assert_eq!(
                middle
                    .exact_boundary_mask(a, &physical, &indices)?
                    .try_to_vec::<f64>()?,
                [1.0; 2]
            );

            let declared_values: Vec<_> = CASES
                .iter()
                .map(|case| i64::try_from(case.chart.owner))
                .collect::<std::result::Result<_, _>>()
                .map_err(|_| numr::error::Error::InvalidArgument {
                    arg: "owner",
                    reason: "Frozen chart owner exceeds I64. Check the fixture".into(),
                })?;
            assert_eq!(declared_values.len(), 17);
            let declared_owners = Tensor::from_slice(&declared_values, &[17], client.device())?;
            let gathered = table.gather(a, &declared_owners)?;
            let reorder = Tensor::from_slice(&[16_i64, 0, 16, 3, 0], &[5], client.device())?;
            let selected = gathered.select(a, &reorder)?;
            assert!(selected.same_model(&gathered));
            let expected_owners = [16_usize, 0, 16, 3, 0].map(|row| declared_values[row]);
            assert_eq!(selected.owners().try_to_vec::<i64>()?, expected_owners);
            let retained: Vec<_> = [
                selected.areas().high(),
                selected.areas().low(),
                selected.areas().abs_error(),
            ]
            .into_iter()
            .map(Tensor::try_to_vec::<f64>)
            .collect::<numr::error::Result<_>>()?;
            for component in [
                gathered.areas().high(),
                gathered.areas().low(),
                gathered.areas().abs_error(),
                all.areas().high(),
            ] {
                client.copy_into(
                    component,
                    &Tensor::zeros(component.shape(), DType::F64, client.device())?,
                )?;
            }
            client.copy_into(
                gathered.owners(),
                &Tensor::zeros(&[17], DType::I64, client.device())?,
            )?;
            for (component, expected) in [
                selected.areas().high(),
                selected.areas().low(),
                selected.areas().abs_error(),
            ]
            .into_iter()
            .zip(retained)
            {
                assert_eq!(component.try_to_vec::<f64>()?, expected);
            }
            assert_eq!(selected.owners().try_to_vec::<i64>()?, expected_owners);
            let fresh = table.gather(a, &owners)?;
            assert_eq!(
                fresh
                    .exact_boundary_mask(a, &source, &first)?
                    .try_to_vec::<f64>()?,
                [1.0; 5]
            );
            Ok(())
        },
    ))?;
    bank_controls::<R>(&client)
}

fn bank_controls<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let scalar = |value| Tensor::<R>::from_slice(&[value], &[1, 1], client.device());
    let bank = BankInputs {
        left_station: scalar(0.0)?,
        right_station: scalar(1.0)?,
        low_elevation: scalar(0.0)?,
        high_elevation: scalar(3.0)?,
        vertical: scalar(0.0)?,
    };
    let extreme = BankInputs {
        left_station: scalar(-f64::MAX)?,
        right_station: scalar(f64::MAX)?,
        low_elevation: scalar(-f64::MAX)?,
        high_elevation: scalar(f64::MAX)?,
        vertical: scalar(0.0)?,
    };
    let vertical = BankInputs {
        left_station: scalar(2.0)?,
        right_station: scalar(2.0)?,
        low_elevation: scalar(-f64::MAX)?,
        high_elevation: scalar(f64::MAX)?,
        vertical: scalar(1.0)?,
    };
    hydraulic(error::checked::<R, _>(
        client,
        "original bank endpoint and interpolation witnesses",
        |a| {
            let stage = |value| Tensor::<R>::from_slice(&[value], &[1], client.device());
            let valid = stage(1.0)?;
            let interior = bank::station(a, &bank, false, 0, &stage(1.0)?, &valid)?;
            assert!(interior.abs_error().item::<f64>()? > 0.0);
            let endpoint = bank::station(a, &bank, false, 0, &stage(3.0)?, &valid)?;
            assert_eq!(endpoint.high().item::<f64>()?, 1.0);
            assert_eq!(endpoint.low().item::<f64>()?, 0.0);
            assert_eq!(endpoint.abs_error().item::<f64>()?, 0.0);
            for (left, stage_value, expected) in [
                (false, -f64::MAX, -f64::MAX),
                (false, f64::MAX, f64::MAX),
                (true, -f64::MAX, f64::MAX),
                (true, f64::MAX, -f64::MAX),
            ] {
                let result = bank::station(a, &extreme, left, 0, &stage(stage_value)?, &valid)?;
                assert_eq!(result.high().item::<f64>()?, expected);
                assert_eq!(result.abs_error().item::<f64>()?, 0.0);
            }
            let result = bank::station(a, &vertical, false, 0, &stage(0.0)?, &valid)?;
            assert_eq!(result.high().item::<f64>()?, 2.0);
            assert_eq!(result.abs_error().item::<f64>()?, 0.0);
            let inactive = bank::station(a, &extreme, false, 0, &stage(0.0)?, &stage(0.0)?)?;
            assert_eq!(inactive.high().item::<f64>()?, 0.0);
            assert_eq!(inactive.abs_error().item::<f64>()?, 0.0);
            Ok(())
        },
    ))
}

#[test]
fn cpu_original_boundary_areas_retain_physical_point_identity() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime};
    boundary_matrix::<CpuRuntime>(CpuClient::new(CpuDevice::new()))
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_original_boundary_areas_retain_physical_point_identity() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    boundary_matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
