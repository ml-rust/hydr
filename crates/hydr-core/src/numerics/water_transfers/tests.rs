use super::{AggregationKind, TransferTree, WaterTransferError};
use numr::runtime::RuntimeClient;
use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
use numr::tensor::Tensor;

type TestResult = Result<(), Box<dyn std::error::Error>>;

fn client() -> CpuClient {
    CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None))
}

#[test]
fn water_absorption_rejects_while_momentum_retains_the_rounding_bound() -> TestResult {
    let client = client();
    let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a"), (0, 1, "b")])?;
    let quantum = f64::from_bits(1);
    let values = Tensor::from_slice(&[1.0, quantum], &[2], client.device())?;
    assert!(matches!(
        tree.aggregate(&client, &values, AggregationKind::Water),
        Err(WaterTransferError::Numerical(_))
    ));
    let momentum = tree.aggregate(&client, &values, AggregationKind::Momentum)?;
    assert_eq!(momentum.value.item::<f64>()?, 1.0);
    assert!(momentum.bound.item::<f64>()? >= quantum);
    Ok(())
}

#[test]
fn signed_cancellation_preserves_earlier_absolute_rounding() -> TestResult {
    let client = client();
    let tree = TransferTree::<CpuRuntime>::new_grouped(
        &client,
        1,
        3,
        &[(0, 0, "a"), (0, 1, "b"), (0, 2, "c")],
    )?;
    let quantum = f64::from_bits(1);
    let values = Tensor::from_slice(&[1.0, quantum, -1.0], &[3], client.device())?;
    let momentum = tree.aggregate(&client, &values, AggregationKind::Momentum)?;
    assert_eq!(momentum.value.item::<f64>()?, 0.0);
    assert!(momentum.bound.item::<f64>()? >= quantum);
    Ok(())
}

#[test]
fn identity_order_preserves_assembly_bits_and_empty_groups() -> TestResult {
    let client = client();
    let quantum = 3.0 * f64::EPSILON / 2.0;
    let original = TransferTree::<CpuRuntime>::new_grouped(
        &client,
        3,
        3,
        &[(1, 0, "b"), (1, 1, "a"), (1, 2, "c")],
    )?;
    let reordered = TransferTree::<CpuRuntime>::new_grouped(
        &client,
        3,
        3,
        &[(1, 2, "a"), (1, 0, "c"), (1, 1, "b")],
    )?;
    let values = Tensor::from_slice(&[quantum, 1.0, 0.0], &[3], client.device())?;
    let reordered_values = Tensor::from_slice(&[0.0, quantum, 1.0], &[3], client.device())?;
    let a = original.aggregate(&client, &values, AggregationKind::Water)?;
    let b = reordered.aggregate(&client, &reordered_values, AggregationKind::Water)?;
    let a_values = a.value.try_to_vec::<f64>()?;
    let b_values = b.value.try_to_vec::<f64>()?;
    assert_eq!(
        a_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        b_values.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(a.bound.try_to_vec::<f64>()?, b.bound.try_to_vec::<f64>()?);
    assert_eq!(a_values[0], 0.0);
    assert_eq!(a_values[2], 0.0);
    assert_eq!(a.bound.try_to_vec::<f64>()?[0], 0.0);
    assert_eq!(
        original.degree_margin.try_to_vec::<f64>()?,
        vec![8.0 * f64::EPSILON, 14.0 * f64::EPSILON, 8.0 * f64::EPSILON]
    );
    Ok(())
}

#[test]
fn transfer_inputs_require_declared_count_rank_and_dtype() -> TestResult {
    let client = client();
    let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a")])?;
    let wrong_count = Tensor::from_slice(&[1.0_f64], &[1], client.device())?;
    let wrong_rank = Tensor::from_slice(&[1.0_f64, 2.0], &[1, 2], client.device())?;
    let wrong_dtype = Tensor::from_slice(&[1.0_f32, 2.0], &[2], client.device())?;
    for values in [&wrong_count, &wrong_rank, &wrong_dtype] {
        assert!(matches!(
            tree.aggregate(&client, values, AggregationKind::Water),
            Err(WaterTransferError::Input(_))
        ));
    }
    Ok(())
}

#[test]
fn strided_transfer_inputs_ignore_nonfinite_padding() -> TestResult {
    let client = client();
    let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 1, 2, &[(0, 0, "a"), (0, 1, "b")])?;
    let values = Tensor::from_slice(
        &[f64::NAN, 1.0, f64::INFINITY, 2.0],
        &[2, 2],
        client.device(),
    )?
    .narrow(1, 1, 1)?
    .squeeze(Some(1));
    assert!(!values.is_contiguous());
    let result = tree.aggregate(&client, &values, AggregationKind::Water)?;
    assert_eq!(result.value.item::<f64>()?, 3.0);
    assert_eq!(result.bound.item::<f64>()?, 0.0);
    Ok(())
}

#[test]
fn empty_input_layout_keeps_all_groups_zero() -> TestResult {
    let client = client();
    let tree = TransferTree::<CpuRuntime>::new_grouped(&client, 3, 0, &[])?;
    let values = Tensor::from_slice(&[] as &[f64], &[0], client.device())?;
    let result = tree.aggregate(&client, &values, AggregationKind::Water)?;
    assert_eq!(result.value.try_to_vec::<f64>()?, vec![0.0; 3]);
    assert_eq!(result.bound.try_to_vec::<f64>()?, vec![0.0; 3]);
    Ok(())
}
