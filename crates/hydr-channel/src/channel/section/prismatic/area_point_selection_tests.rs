use super::area_point_query::PrismaticAreaPointRecovery;
use super::area_query::test_support as support;
use super::config::PrismaticRecoveryConfig;
use crate::channel::flow::surveyed::error::{self, HydraulicError};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use support::{Client, Result, hydraulic};

fn tensors<R: Runtime<DType = DType>>(source: &PrismaticAreaPointRecovery<R>) -> Vec<&Tensor<R>> {
    let coordinates = source.band_coordinates();
    let mut values = vec![
        source.requested_areas_m2().high(),
        source.requested_areas_m2().low(),
        source.requested_areas_m2().abs_error(),
        source.dry_mask(),
        coordinates.candidate_mask(),
    ];
    for value in [
        source.depth_m(),
        source.forward_area_m2(),
        source.top_width_m(),
        source.hydrostatic_integral_m3(),
        source.hydraulic_depth_m(),
        coordinates.area_above_start_m2(),
        coordinates.area_below_end_m2(),
        coordinates.depth_above_start_m(),
        coordinates.depth_below_end_m(),
    ] {
        values.extend([value.lower(), value.upper()]);
    }
    values
}

fn snapshot<R: Runtime<DType = DType>>(
    source: &PrismaticAreaPointRecovery<R>,
) -> Result<Vec<Vec<u64>>> {
    tensors(source)
        .into_iter()
        .map(|tensor| {
            Ok(tensor
                .contiguous()?
                .try_to_vec::<f64>()?
                .into_iter()
                .map(f64::to_bits)
                .collect())
        })
        .collect()
}

fn selection<R>(client: &R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let section = support::batch::<R>(client)?;
    let foreign = support::batch::<R>(client)?;
    let owner_storage = Tensor::from_slice(&[99_i64, 0, 99, 1, 99, 4], &[3, 2], client.device())?;
    let owners = owner_storage.narrow(1, 1, 1)?.squeeze(Some(1));
    assert!(!owners.is_contiguous());
    let areas = support::expansion::<R>(
        client,
        &[
            [0x3feccccccccccccd, 0xbc80000000000000, 0],
            [0, 0, 0],
            [2.0_f64.to_bits(), 0, 0],
        ],
        true,
    )?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let other_source =
        foreign.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let before = snapshot(&source)?;
    let properties = source.properties()?;
    let row_storage =
        Tensor::from_slice(&[99_i64, 2, 99, 0, 99, 2, 99, 1], &[4, 2], client.device())?;
    let rows = row_storage.narrow(1, 1, 1)?.squeeze(Some(1));
    assert!(!rows.is_contiguous());
    hydraulic(error::checked::<R, _>(
        client,
        "owned source recovery row selection",
        |a| {
            let selected = source.select(a, &rows)?;
            assert!(selected.same_context(&source.context));
            assert!(!selected.same_context(&other_source.context));
            assert_eq!(selected.owners().try_to_vec::<i64>()?, [4, 0, 4, 1]);
            assert_eq!(
                selected.dry_mask().try_to_vec::<f64>()?,
                [0.0, 0.0, 0.0, 1.0]
            );
            let actual = snapshot(&selected).map_err(|error| HydraulicError::Numerical {
                context: error.to_string(),
            })?;
            for (field, (expected, actual)) in before.iter().zip(&actual).enumerate() {
                let width = expected.len() / 3;
                let expected: Vec<_> = [2, 0, 2, 1]
                    .into_iter()
                    .flat_map(|row| expected[row * width..(row + 1) * width].iter().copied())
                    .collect();
                assert_eq!(*actual, expected, "stored field={field}");
            }
            assert_eq!(
                selected.properties()?,
                [
                    properties[2].clone(),
                    properties[0].clone(),
                    properties[2].clone(),
                    properties[1].clone()
                ]
            );
            let empty_rows = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
            let empty = source.select(a, &empty_rows)?;
            assert!(empty.same_context(&source.context));
            assert_eq!(empty.owners().shape(), &[0]);
            assert_eq!(
                empty.band_coordinates().candidate_mask().shape(),
                &[0, source.context.maximum_bands]
            );
            for tensor in tensors(&empty) {
                assert_eq!(tensor.numel(), 0);
            }
            let wrong_type = Tensor::zeros(&[1], DType::F64, client.device())?;
            assert!(source.select(a, &wrong_type).is_err());
            let wrong_rank = Tensor::zeros(&[1, 1], DType::I64, client.device())?;
            assert!(source.select(a, &wrong_rank).is_err());
            for indices in [[-1_i64], [3]] {
                let invalid = Tensor::from_slice(&indices, &[1], client.device())?;
                assert!(source.select(a, &invalid).is_err());
            }
            // Selection copies stored certificates. A synthetic radius checks preservation without invoking recovery.
            let radius = Tensor::from_slice(
                &[f64::EPSILON, 0.0, f64::MIN_POSITIVE],
                &[3],
                client.device(),
            )?;
            client.copy_into(source.requested_areas_m2().abs_error(), &radius)?;
            let uncertain = source.select(a, &rows)?;
            assert_eq!(
                uncertain
                    .requested_areas_m2()
                    .abs_error()
                    .try_to_vec::<f64>()?,
                [f64::MIN_POSITIVE, f64::EPSILON, f64::MIN_POSITIVE, 0.0]
            );
            assert_eq!(
                uncertain.depth_m().lower().try_to_vec::<f64>()?,
                selected.depth_m().lower().try_to_vec::<f64>()?
            );
            for tensor in tensors(&source) {
                client.copy_into(
                    tensor,
                    &Tensor::zeros(tensor.shape(), DType::F64, client.device())?,
                )?;
            }
            client.copy_into(
                source.owners(),
                &Tensor::zeros(source.owners().shape(), DType::I64, client.device())?,
            )?;
            client.copy_into(
                &row_storage,
                &Tensor::zeros(row_storage.shape(), DType::I64, client.device())?,
            )?;
            assert_eq!(
                snapshot(&selected).map_err(|error| HydraulicError::Numerical {
                    context: error.to_string()
                })?,
                actual
            );
            assert_eq!(selected.owners().try_to_vec::<i64>()?, [4, 0, 4, 1]);
            assert_eq!(
                uncertain
                    .requested_areas_m2()
                    .abs_error()
                    .try_to_vec::<f64>()?,
                [f64::MIN_POSITIVE, f64::EPSILON, f64::MIN_POSITIVE, 0.0]
            );
            Ok(())
        },
    ))
}

#[test]
fn configured_cpu_source_recovery_selection_preserves_all_stored_certificates() -> Result {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let original = CpuClient::new(CpuDevice::new());
    let before = original.parallelism();
    let source_client = original
        .clone()
        .with_parallelism(ParallelismConfig::new(Some(1), None));
    let section = support::batch::<CpuRuntime>(&source_client)?;
    let owners = Tensor::from_slice(&[1_i64], &[1], source_client.device())?;
    let areas =
        support::expansion::<CpuRuntime>(&source_client, &[[1.0_f64.to_bits(), 0, 0]], false)?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let config = ParallelismConfig::new(Some(2), Some(257));
    let caller = original.clone().with_parallelism(config);
    hydraulic(error::checked::<CpuRuntime, _>(
        &caller,
        "same-device configured source selection",
        |a| {
            let selected =
                source.select(a, &Tensor::from_slice(&[0_i64], &[1], caller.device())?)?;
            assert!(selected.same_context(&source.context));
            Ok(())
        },
    ))?;
    selection::<CpuRuntime>(&caller)?;
    assert_eq!(caller.parallelism(), config);
    assert_eq!(original.parallelism(), before);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_source_recovery_selection_preserves_all_stored_certificates() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    selection::<CudaRuntime>(&client)?;
    let section = support::batch::<CudaRuntime>(&client)?;
    let owners = Tensor::from_slice::<i64>(&[], &[0], client.device())?;
    let areas = support::expansion::<CudaRuntime>(&client, &[], false)?;
    let source =
        section.recover_area_points_tensor(&owners, &areas, &PrismaticRecoveryConfig::default())?;
    let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &CudaDevice::new(usize::MAX))?;
    hydraulic(error::checked::<CudaRuntime, _>(
        &client,
        "foreign source selection indices",
        |a| {
            assert!(source.select(a, &foreign).is_err());
            Ok(())
        },
    ))?;
    if CudaDevice::count()? > 1 {
        let other = CudaClient::new(CudaDevice::new(1))?;
        hydraulic(error::checked::<CudaRuntime, _>(
            &other,
            "foreign source selection caller",
            |a| {
                let rows = Tensor::from_slice::<i64>(&[], &[0], other.device())?;
                assert!(source.select(a, &rows).is_err());
                Ok(())
            },
        ))?;
    }
    Ok(())
}
