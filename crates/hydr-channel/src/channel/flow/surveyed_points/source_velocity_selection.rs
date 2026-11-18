use super::{CapturedPointInput, SourceVelocity};
use crate::channel::{
    flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult},
    section::CrossSectionError,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps,
    },
    runtime::{Device, Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

impl<R> SourceVelocity<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Select captured sources and their existing certificates through one owned row tensor.
    pub(crate) fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let count = self.input.owners.numel();
        metadata(a, rows, rows.numel(), DType::I64)?;
        metadata(a, &self.input.owners, count, DType::I64)?;
        for value in [
            &self.input.left_area,
            &self.input.left_discharge,
            &self.input.right_area,
            &self.input.right_discharge,
            &self.left,
            &self.right,
            &self.difference,
        ] {
            for component in [value.high(), value.low(), value.abs_error()] {
                metadata(a, component, count, a.dtype())?;
            }
        }
        for mask in [
            &self.input.left_dry,
            &self.input.right_dry,
            &self.both_wet,
            &self.uniform,
        ] {
            metadata(a, mask, count, a.dtype())?;
        }
        u32::try_from(rows.numel()).map_err(|_| {
            invalid(format!(
                "source selection count={} exceeds U32 indexing. Split the selection",
                rows.numel()
            ))
        })?;
        let limit = i64::try_from(count).map_err(|_| {
            invalid(format!(
                "source row count={count} exceeds I64 indexing. Split the source batch"
            ))
        })?;
        let captured = Tensor::empty(rows.shape(), DType::I64, c.device())?;
        c.copy_into(&captured, rows)?;
        if captured.numel() != 0 {
            let zero = Tensor::from_slice(&[0_i64], &[], c.device())?;
            let upper = Tensor::from_slice(&[limit], &[], c.device())?;
            let bad = c.maximum(&c.lt(&captured, &zero)?, &c.ge(&captured, &upper)?)?;
            let bad = c.nonzero(&bad)?.flatten()?;
            if bad.numel() != 0 {
                let first = bad.narrow(0, 0, 1)?;
                let value = c.index_select(&captured, 0, &first)?.item::<i64>()?;
                return Err(invalid(format!(
                    "source selection row={value} lies outside 0..{count}. Supply an admitted source row"
                )));
            }
        }
        let select = |value: &Tensor<R>| -> HydraulicResult<Tensor<R>> {
            Ok(c.index_select(value, 0, &captured)?)
        };
        let input = Arc::new(CapturedPointInput {
            owners: select(&self.input.owners)?,
            left_area: expansion(a, &self.input.left_area, &captured)?,
            left_discharge: expansion(a, &self.input.left_discharge, &captured)?,
            right_area: expansion(a, &self.input.right_area, &captured)?,
            right_discharge: expansion(a, &self.input.right_discharge, &captured)?,
            left_dry: select(&self.input.left_dry)?,
            right_dry: select(&self.input.right_dry)?,
        });
        Ok(Self {
            input,
            left: expansion(a, &self.left, &captured)?,
            right: expansion(a, &self.right, &captured)?,
            difference: expansion(a, &self.difference, &captured)?,
            both_wet: select(&self.both_wet)?,
            uniform: select(&self.uniform)?,
        })
    }
}

fn metadata<R>(
    a: &Arithmetic<'_, R>,
    value: &Tensor<R>,
    count: usize,
    dtype: DType,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    if value.shape() != [count]
        || value.dtype() != dtype
        || !value.device().is_same(a.client().device())
    {
        return Err(invalid(format!(
            "source selection metadata shape={:?}, dtype={:?}, device={}: supply shape=[{count}], dtype={dtype:?}, device={}",
            value.shape(),
            value.dtype(),
            value.device().id(),
            a.client().device().id()
        )));
    }
    Ok(())
}

fn expansion<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    Ok(a.expansion_bounds(
        &c.index_select(value.high(), 0, rows)?,
        &c.index_select(value.low(), 0, rows)?,
        &c.index_select(value.abs_error(), 0, rows)?,
    )?)
}

fn invalid(context: String) -> HydraulicError {
    CrossSectionError::InvalidInput {
        section: None,
        context,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use crate::channel::flow::surveyed_points::source_fixture_support::{
        self as support, Client, Result, batch, components, expansion, hydraulic,
    };
    use crate::channel::flow::surveyed_points::{source_velocity, types::SourceVelocity};
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
        let indices_storage =
            Tensor::from_slice(&indices_storage, &[order.len(), 2], client.device())?;
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
}
