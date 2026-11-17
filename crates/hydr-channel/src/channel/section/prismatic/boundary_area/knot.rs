use super::super::inputs::{Inputs, column};
use super::bank;
use crate::channel::flow::surveyed::error::Arithmetic;
use numr::{
    dtype::DType,
    ops::{CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{ArithmeticResult, CertifiedExpansion};

/// Two-part band quantities from original coordinates, one value per section row.
///
/// Invalid band columns hold zero widths, height, and increment.
pub(in crate::channel::section::prismatic) struct BandKnot<R: Runtime<DType = DType>> {
    /// Wet width at the band bottom, in metres.
    pub start_width: CertifiedExpansion<R>,
    /// Band top minus band bottom, in metres.
    pub height: CertifiedExpansion<R>,
    /// Mean of the bottom and top widths, in metres.
    pub mean_width: CertifiedExpansion<R>,
    /// Band area, height times mean width, in square metres.
    pub increment: CertifiedExpansion<R>,
}

/// Evaluate one band column. `zeros` has shape `[rows]` and `half` holds 0.5 in that shape.
pub(in crate::channel::section::prismatic) fn band<R>(
    a: &mut Arithmetic<'_, R>,
    inputs: &Inputs<R>,
    band: usize,
    zeros: &Tensor<R>,
    half: &CertifiedExpansion<R>,
) -> ArithmeticResult<BandKnot<R>>
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
    let rows = zeros.numel();
    let valid = column(&inputs.valid, band, rows)?;
    let bottom = column(&inputs.band_bottom, band, rows)?;
    let top = column(&inputs.band_top, band, rows)?;
    let left_start = bank::station(a, &inputs.left_bank, true, band, &bottom, &valid)?;
    let right_start = bank::station(a, &inputs.right_bank, false, band, &bottom, &valid)?;
    let left_end = bank::station(a, &inputs.left_bank, true, band, &top, &valid)?;
    let right_end = bank::station(a, &inputs.right_bank, false, band, &top, &valid)?;
    let start_width = a.expansion_sub(&right_start, &left_start)?;
    let end_width = a.expansion_sub(&right_end, &left_end)?;
    let bottom_point = a.expansion_point(&c.where_cond(&valid, &bottom, zeros)?)?;
    let top_point = a.expansion_point(&c.where_cond(&valid, &top, zeros)?)?;
    let height = a.expansion_sub(&top_point, &bottom_point)?;
    // Scale each width separately to avoid an overflowing width sum.
    let half_start = a.expansion_mul(&start_width, half)?;
    let half_end = a.expansion_mul(&end_width, half)?;
    let mean_width = a.expansion_add(&half_start, &half_end)?;
    let increment = a.expansion_mul(&height, &mean_width)?;
    Ok(BandKnot {
        start_width,
        height,
        mean_width,
        increment,
    })
}
