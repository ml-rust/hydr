use super::super::{band_terms::stack, config, context::Context};
use super::{BoundaryAreaTable, knot};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
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
            let band = knot::band(a, inputs, band, &zeros, &half)?;
            prefix = a.expansion_add(&prefix, &band.increment)?;
            columns.push(a.expansion_bounds(prefix.high(), prefix.low(), prefix.abs_error())?);
        }
        let areas = stack(a, &columns)?;
        let cap_indices = c.cast(&c.sum(&inputs.valid, &[0], false)?, DType::I64)?;
        Ok(Self {
            model: Arc::clone(model),
            areas,
            cap_indices,
        })
    }
}
