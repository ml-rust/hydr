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
