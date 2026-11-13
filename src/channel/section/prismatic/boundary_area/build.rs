use super::super::{config, context::Context, inputs::column};
use super::{BoundaryAreaTable, bank};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, ConditionalOps, NextRepresentableOps, ReduceOps, ScalarOps, ShapeOps,
        TensorOps, TwoSumOps, TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;
use std::sync::Arc;

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
    pub(in crate::channel::section::prismatic) fn build(
        a: &mut Arithmetic<'_, R>,
        model: &Arc<Context<R>>,
    ) -> HydraulicResult<Self> {
        let rows = model.source.definitions.len();
        let boundaries = model.maximum_bands.checked_add(1).ok_or_else(|| {
            numr::error::Error::InvalidArgument {
                arg: "boundary_area",
                reason: format!(
                    "band_count={} overflows boundary count. Split the section batch",
                    model.maximum_bands
                ),
            }
        })?;
        config::check_work(rows, boundaries)?;
        let c = a.client();
        let inputs = &model.raw_inputs;
        let zeros = Tensor::zeros(&[rows], DType::F64, c.device())?;
        let half =
            a.expansion_point(&Tensor::full_scalar(&[rows], DType::F64, 0.5, c.device())?)?;
        let mut prefix = a.expansion_point(&zeros)?;
        let mut columns = Vec::with_capacity(boundaries);
        columns.push(a.expansion_point(&zeros)?);
        for band in 0..model.maximum_bands {
            let valid = column(&inputs.valid, band, rows)?;
            let bottom = column(&inputs.band_bottom, band, rows)?;
            let top = column(&inputs.band_top, band, rows)?;
            let left_start = bank::station(a, &inputs.left_bank, true, band, &bottom, &valid)?;
            let right_start = bank::station(a, &inputs.right_bank, false, band, &bottom, &valid)?;
            let left_end = bank::station(a, &inputs.left_bank, true, band, &top, &valid)?;
            let right_end = bank::station(a, &inputs.right_bank, false, band, &top, &valid)?;
            let start_width = a.expansion_sub(&right_start, &left_start)?;
            let end_width = a.expansion_sub(&right_end, &left_end)?;
            let bottom = a.expansion_point(&c.where_cond(&valid, &bottom, &zeros)?)?;
            let top = a.expansion_point(&c.where_cond(&valid, &top, &zeros)?)?;
            let height = a.expansion_sub(&top, &bottom)?;
            // Scale each width separately to avoid an overflowing width sum.
            let half_start = a.expansion_mul(&start_width, &half)?;
            let half_end = a.expansion_mul(&end_width, &half)?;
            let mean_width = a.expansion_add(&half_start, &half_end)?;
            let increment = a.expansion_mul(&height, &mean_width)?;
            prefix = a.expansion_add(&prefix, &increment)?;
            columns.push(a.expansion_bounds(prefix.high(), prefix.low(), prefix.abs_error())?);
        }
        let stack =
            |component: fn(&CertifiedExpansion<R>) -> &Tensor<R>| -> HydraulicResult<Tensor<R>> {
                Ok(c.stack(&columns.iter().map(component).collect::<Vec<_>>(), 1)?)
            };
        let areas = a.expansion_bounds(
            &stack(CertifiedExpansion::high)?,
            &stack(CertifiedExpansion::low)?,
            &stack(CertifiedExpansion::abs_error)?,
        )?;
        let cap_indices = c.cast(&c.sum(&inputs.valid, &[0], false)?, DType::I64)?;
        Ok(Self {
            model: Arc::clone(model),
            areas,
            cap_indices,
        })
    }
}
