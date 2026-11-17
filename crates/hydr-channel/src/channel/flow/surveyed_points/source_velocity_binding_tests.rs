use super::source_tests::support::{self, Client, Result, batch, components, expansion, hydraulic};
use super::{source_velocity, types::SourceVelocity};
use crate::channel::{
    flow::surveyed::error::{self, HydraulicError},
    section::CrossSectionError,
};
use numr::{
    dtype::DType,
    ops::BinaryOps,
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use std::sync::Arc;

#[derive(Debug, PartialEq)]
struct BoundRow {
    owner: i64,
    captured: [[u64; 3]; 4],
    dry: [u64; 2],
    derived: [[u64; 3]; 3],
    masks: [u64; 2],
}

fn snapshot<R: Runtime<DType = DType>>(velocity: &SourceVelocity<R>) -> Result<Vec<BoundRow>> {
    let input = velocity.input();
    let owners = input.owners.try_to_vec::<i64>()?;
    let captured = [
        &input.left_area,
        &input.left_discharge,
        &input.right_area,
        &input.right_discharge,
    ]
    .into_iter()
    .map(components)
    .collect::<Result<Vec<_>>>()?;
    let derived = [&velocity.left, &velocity.right, &velocity.difference]
        .into_iter()
        .map(components)
        .collect::<Result<Vec<_>>>()?;
    let dry = [&input.left_dry, &input.right_dry]
        .into_iter()
        .map(|v| v.try_to_vec::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let masks = [&velocity.both_wet, &velocity.uniform]
        .into_iter()
        .map(|v| v.try_to_vec::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(owners
        .into_iter()
        .enumerate()
        .map(|(row, owner)| BoundRow {
            owner,
            captured: std::array::from_fn(|field| captured[field][row]),
            derived: std::array::from_fn(|field| derived[field][row]),
            dry: std::array::from_fn(|side| dry[side][row].to_bits()),
            masks: std::array::from_fn(|field| masks[field][row].to_bits()),
        })
        .collect())
}

fn select<R>(
    client: &R::Client,
    velocity: &SourceVelocity<R>,
    rows: &Tensor<R>,
) -> error::HydraulicResult<SourceVelocity<R>>
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    error::checked::<R, _>(client, "bound velocity selection", |a| {
        velocity.select(a, rows)
    })
}

fn invalid<T>(result: error::HydraulicResult<T>) -> Result {
    let error = result
        .err()
        .ok_or("Bound velocity selection accepts invalid rows")?;
    assert!(
        matches!(
            error,
            HydraulicError::Geometry(CrossSectionError::InvalidInput { .. })
        ),
        "{error:?}"
    );
    Ok(())
}

fn matrix<R>(client: R::Client) -> Result
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let cases = support::cases();
    let section = batch::<R>(&client)?;
    let owner_values = cases
        .iter()
        .map(|case| i64::try_from(case.owner))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let owner_storage = owner_values
        .iter()
        .flat_map(|&owner| [i64::MAX, owner])
        .collect::<Vec<_>>();
    let owners = Tensor::from_slice(&owner_storage, &[cases.len(), 2], client.device())?
        .narrow(1, 1, 1)?
        .squeeze(Some(1));
    assert!(!owners.is_contiguous() && owners.offset() > 0);
    let inputs = (0..4)
        .map(|field| {
            expansion::<R>(
                &client,
                &cases
                    .iter()
                    .map(|case| case.inputs[field])
                    .collect::<Vec<_>>(),
                true,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let captured = hydraulic(
        section.capture_riemann_points(&owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3]),
    )?;
    let weak = Arc::downgrade(&captured);
    let velocity = hydraulic(source_velocity::prepare::<R>(&client, &captured))?;
    let repeated = hydraulic(source_velocity::prepare::<R>(&client, &captured))?;
    assert!(std::ptr::eq(velocity.input(), captured.as_ref()));
    assert!(std::ptr::eq(velocity.input(), repeated.input()));
    let expected = snapshot(&velocity)?;
    assert_eq!(expected, snapshot(&repeated)?);
    assert!(
        expected
            .iter()
            .any(|row| row.dry.iter().any(|&dry| f64::from_bits(dry) == 1.0))
    );
    assert!(expected.iter().any(|row| row.masks[0] == 1.0_f64.to_bits()));
    assert!(expected.iter().any(|row| row.masks[1] == 1.0_f64.to_bits()));
    let radius_row = expected
        .iter()
        .position(|row| {
            row.derived
                .iter()
                .any(|point| f64::from_bits(point[2]) > 0.0)
        })
        .ok_or("Frozen source corpus has no nonzero derived radius")?;
    let dry_row = expected
        .iter()
        .position(|row| row.dry.iter().any(|&v| f64::from_bits(v) == 1.0))
        .ok_or("Frozen source corpus has no dry row")?;
    for value in inputs
        .iter()
        .flat_map(|point| [point.high(), point.low(), point.abs_error()])
        .chain(std::iter::once(&owners))
    {
        client.copy_into(
            value,
            &Tensor::zeros(value.shape(), value.dtype(), client.device())?,
        )?;
    }
    assert_eq!(expected, snapshot(&velocity)?);
    drop(repeated);
    drop(captured);
    assert!(weak.upgrade().is_some());

    let order = [dry_row, radius_row, dry_row, 0];
    let index_values = order
        .into_iter()
        .map(i64::try_from)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let indices_storage = index_values
        .iter()
        .flat_map(|&row| [i64::MIN, row])
        .collect::<Vec<_>>();
    let indices_storage = Tensor::from_slice(&indices_storage, &[order.len(), 2], client.device())?;
    let indices = indices_storage.narrow(1, 1, 1)?.squeeze(Some(1));
    assert!(!indices.is_contiguous() && indices.offset() > 0);
    let selected = hydraulic(select::<R>(&client, &velocity, &indices))?;
    assert!(!std::ptr::eq(selected.input(), velocity.input()));
    assert!(selected.input().owners.device().is_same(client.device()));
    assert!(selected.left.high().device().is_same(client.device()));
    let selected_before = snapshot(&selected)?;
    for (row, &original) in order.iter().enumerate() {
        assert_eq!(selected_before[row], expected[original]);
    }
    client.copy_into(
        &indices_storage,
        &Tensor::zeros(indices_storage.shape(), DType::I64, client.device())?,
    )?;
    assert_eq!(selected_before, snapshot(&selected)?);
    assert_eq!(expected, snapshot(&velocity)?);
    let nested_rows = Tensor::from_slice(&[3_i64, 1, 1, 0], &[4], client.device())?;
    let nested = hydraulic(select::<R>(&client, &selected, &nested_rows))?;
    assert!(!std::ptr::eq(nested.input(), selected.input()));
    let nested_before = snapshot(&nested)?;
    for (row, original) in [3_usize, 1, 1, 0].into_iter().enumerate() {
        assert_eq!(nested_before[row], selected_before[original]);
    }

    let empty_rows = Tensor::from_slice(&[] as &[i64], &[0], client.device())?;
    let empty = hydraulic(select::<R>(&client, &velocity, &empty_rows))?;
    assert!(snapshot(&empty)?.is_empty());
    assert!(!std::ptr::eq(empty.input(), velocity.input()));
    assert!(snapshot(&hydraulic(select::<R>(&client, &empty, &empty_rows))?)?.is_empty());
    for rows in [
        Tensor::from_slice(&[-1_i64], &[1], client.device())?,
        Tensor::from_slice(&[i64::try_from(cases.len())?], &[1], client.device())?,
        Tensor::from_slice(&[0_i64], &[], client.device())?,
        Tensor::from_slice(&[0_i64], &[1, 1], client.device())?,
    ] {
        invalid(select::<R>(&client, &velocity, &rows))?;
    }
    invalid(select::<R>(
        &client,
        &velocity,
        &Tensor::zeros(&[1], velocity.left.high().dtype(), client.device())?,
    ))?;
    invalid(select::<R>(
        &client,
        &empty,
        &Tensor::from_slice(&[0_i64], &[1], client.device())?,
    ))?;
    let row = Tensor::from_slice(&[0_i64], &[1], client.device())?;
    let mut malformed = hydraulic(select::<R>(&client, &velocity, &row))?;
    malformed.left = expansion::<R>(&client, &[[0; 3]; 2], false)?;
    invalid(select::<R>(&client, &malformed, &row))?;

    for tensor in [
        selected.input().left_area.high(),
        selected.input().left_discharge.low(),
        selected.input().right_area.high(),
        selected.input().right_discharge.abs_error(),
        &selected.input().owners,
        selected.left.high(),
    ] {
        client.copy_into(
            tensor,
            &Tensor::zeros(tensor.shape(), tensor.dtype(), client.device())?,
        )?;
    }
    assert_eq!(nested_before, snapshot(&nested)?);
    assert_eq!(expected, snapshot(&velocity)?);
    drop(velocity);
    assert!(weak.upgrade().is_none());
    Ok(())
}

#[test]
fn configured_cpu_source_velocity_binding_preserves_owned_selection() -> Result {
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
fn cuda_source_velocity_binding_preserves_owned_selection() -> Result {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
}
