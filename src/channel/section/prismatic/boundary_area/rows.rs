use super::super::config;
use super::{BoundaryAreaTable, BoundaryAreas};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

fn select_expansion<R>(
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

impl<R> BoundaryAreaTable<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn gather(
        &self,
        a: &mut Arithmetic<'_, R>,
        owners: &Tensor<R>,
    ) -> HydraulicResult<BoundaryAreas<R>> {
        config::check_work(owners.numel(), self.areas.high().shape()[1])?;
        let c = a.client();
        let captured = Tensor::empty(owners.shape(), owners.dtype(), c.device())?;
        c.copy_into(&captured, owners)?;
        let areas = select_expansion(a, &self.areas, &captured)?;
        let cap_indices = c.index_select(&self.cap_indices, 0, &captured)?;
        Ok(BoundaryAreas {
            model: Arc::clone(&self.model),
            owners: captured,
            areas,
            cap_indices,
        })
    }
}

impl<R: Runtime<DType = DType>> BoundaryAreas<R> {
    pub(in crate::channel::section::prismatic) fn same_context(
        &self,
        context: &Arc<super::super::context::Context<R>>,
    ) -> bool {
        Arc::ptr_eq(&self.model, context)
    }

    pub(crate) fn areas(&self) -> &CertifiedExpansion<R> {
        &self.areas
    }
    pub(crate) fn owners(&self) -> &Tensor<R> {
        &self.owners
    }
    pub(crate) fn cap_indices(&self) -> &Tensor<R> {
        &self.cap_indices
    }
    pub(crate) fn same_model(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.model, &other.model)
    }
}

impl<R> BoundaryAreas<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    pub(crate) fn select(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<Self> {
        config::check_work(rows.numel(), self.areas.high().shape()[1])?;
        let c = a.client();
        Ok(Self {
            model: Arc::clone(&self.model),
            owners: c.index_select(&self.owners, 0, rows)?,
            areas: select_expansion(a, &self.areas, rows)?,
            cap_indices: c.index_select(&self.cap_indices, 0, rows)?,
        })
    }

    pub(crate) fn boundary(
        &self,
        a: &mut Arithmetic<'_, R>,
        indices: &Tensor<R>,
    ) -> HydraulicResult<CertifiedExpansion<R>> {
        let c = a.client();
        let count = self.owners.numel();
        let indices = indices.reshape(&[count, 1])?;
        Ok(a.expansion_bounds(
            &c.gather(self.areas.high(), 1, &indices)?
                .reshape(&[count])?,
            &c.gather(self.areas.low(), 1, &indices)?.reshape(&[count])?,
            &c.gather(self.areas.abs_error(), 1, &indices)?
                .reshape(&[count])?,
        )?)
    }

    pub(crate) fn exact_boundary_mask(
        &self,
        a: &mut Arithmetic<'_, R>,
        source: &CertifiedExpansion<R>,
        indices: &Tensor<R>,
    ) -> HydraulicResult<Tensor<R>> {
        let boundary = self.boundary(a, indices)?;
        let difference = a.expansion_sub(source, &boundary)?;
        let c = a.client();
        let zero = Tensor::zeros(source.high().shape(), DType::F64, c.device())?;
        Ok(c.mul(
            &c.eq(difference.high(), &zero)?,
            &c.mul(
                &c.eq(difference.low(), &zero)?,
                &c.eq(difference.abs_error(), &zero)?,
            )?,
        )?)
    }
}
