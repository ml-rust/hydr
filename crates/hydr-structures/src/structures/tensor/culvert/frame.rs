//! Each barrel's frame: the barrel scaled by the exact power of two `2^-e` that takes its rise
//! into `[1, 2)`.
//!
//! No frame product underflows or overflows. A metre length is its frame length times `2^e`,
//! an area times `2^2e` and a moment times `2^3e`, each exact unless it leaves the normal F64
//! range. So a barrel and its power-of-two multiples share one frame and answer alike.
//!
//! Arc admission guarantees that every arc barrel scales exactly into its frame. A circle always
//! does. A box whose framed span would leave the normal range or fail admission keeps its metre
//! dimensions, with `e = 0`.

use super::context::{Bounds, TensorBarrelContext};
use crate::structures::numerics::culvert::BarrelClient;
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Rows of the framed barrels with the scale of each row.
pub(crate) struct BarrelFrame<R: Runtime<DType = DType>> {
    /// The framed barrels. Point geometry reads it. Its depth certificates stay unfilled.
    pub(crate) context: TensorBarrelContext<R>,
    /// `2^e` per row.
    pub(crate) scale: Tensor<R>,
    /// `2^-e` per row.
    pub(crate) unscale: Tensor<R>,
}

impl<R: Runtime<DType = DType>> BarrelFrame<R> {
    /// The frame rows of `framed`, the framed barrels of `barrels`.
    pub(super) fn new(
        client: R::Client,
        barrels: &[CulvertBarrel],
        framed: &[CulvertBarrel],
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let m = barrels.len();
        let ratio = |f: &dyn Fn(&CulvertBarrel, &CulvertBarrel) -> f64| -> Vec<f64> {
            barrels.iter().zip(framed).map(|(b, x)| f(b, x)).collect()
        };
        // Both rises are normal and differ by an exact power of two.
        let scale = ratio(&|b, x| b.rise_m() / x.rise_m());
        let unscale = ratio(&|b, x| x.rise_m() / b.rise_m());
        let context = TensorBarrelContext::rows(client, framed, None)?;
        let device = &context.device;
        Ok(Self {
            scale: Tensor::<R>::from_slice(&scale, &[m], device)?,
            unscale: Tensor::<R>::from_slice(&unscale, &[m], device)?,
            context,
        })
    }

    /// Gathers the rows at the I64 indices `rows`, in their order.
    pub(crate) fn select(&self, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = &self.context.client;
        Ok(Self {
            context: self.context.select(rows)?,
            scale: c.index_select(&self.scale, 0, rows)?,
            unscale: c.index_select(&self.unscale, 0, rows)?,
        })
    }

    /// A frame enclosure of length power `power` back in metres: each bound times `2^e`,
    /// `power` times, rounded outward.
    ///
    /// Each product by a power of two is exact unless it leaves the normal range. Multiplying
    /// back by `2^-e` detects a rounded product, and the bound then steps one F64 outward.
    pub(crate) fn times(
        &self,
        [lower, upper]: [&Tensor<R>; 2],
        power: usize,
    ) -> numr::error::Result<Bounds<R>>
    where
        R::Client: BarrelClient<R>,
    {
        let c = &self.context.client;
        let mut lower = lower.clone();
        let mut upper = upper.clone();
        for _ in 0..power {
            let low = c.mul(&lower, &self.scale)?;
            let high = c.mul(&upper, &self.scale)?;
            let low_up = c.gt(&c.mul(&low, &self.unscale)?, &lower)?;
            let high_down = c.lt(&c.mul(&high, &self.unscale)?, &upper)?;
            lower = c.where_cond(&low_up, &c.next_down(&low)?, &low)?;
            upper = c.where_cond(&high_down, &c.next_up(&high)?, &high)?;
        }
        Ok([lower, upper])
    }
}

/// `2^-e` for the rise `rise_m`: the exact power of two that takes it into `[1, 2)`.
fn unscale_of(rise_m: f64) -> f64 {
    // An admitted rise is a normal F64, so its biased exponent gives the scale exactly.
    let biased = i32::try_from((rise_m.to_bits() >> 52) & 0x7ff).unwrap_or(1023);
    2f64.powi(1023 - biased)
}

/// The barrel in its frame.
///
/// Rejects, naming `row`, an arc barrel whose framed dimensions fail admission.
pub(super) fn framed_barrel(
    row: usize,
    barrel: &CulvertBarrel,
) -> Result<CulvertBarrel, StructureError> {
    let shape = barrel.shape();
    let scale = unscale_of(barrel.rise_m());
    let span = barrel.span_m() * scale;
    let rise = barrel.rise_m() * scale;
    let given: Vec<f64> = barrel
        .given_radius_m()
        .unwrap_or(&[])
        .iter()
        .map(|radius| radius * scale)
        .collect();
    let scaled = match (shape, given.as_slice()) {
        (CulvertBarrelShape::Circular, &[]) => CulvertBarrel::circular(rise),
        (CulvertBarrelShape::Box, &[]) => {
            let exact =
                span >= f64::MIN_POSITIVE && span.is_finite() && span / scale == barrel.span_m();
            let framed = CulvertBarrel::rectangular_box(span, rise);
            return Ok(framed.ok().filter(|_| exact).unwrap_or(*barrel));
        }
        (CulvertBarrelShape::HorizontalEllipse, &[side]) => {
            CulvertBarrel::horizontal_ellipse(span, rise, side)
        }
        (CulvertBarrelShape::VerticalEllipse, &[end]) => {
            CulvertBarrel::vertical_ellipse(span, rise, end)
        }
        (CulvertBarrelShape::PipeArch, &[corner, bottom]) => {
            CulvertBarrel::pipe_arch(span, rise, corner, bottom)
        }
        (CulvertBarrelShape::Arch, &[]) => CulvertBarrel::arch(span, rise),
        _ => Err(StructureError::definition("a barrel lost its given radii")),
    };
    scaled.map_err(|error| {
        StructureError::definition(format!(
            "shape={shape:?},span_m={:?},rise_m={:?}: the barrel context scales the barrel by a power of two to a rise in [1, 2), and the scaled barrel fails admission ({error}). Use dimensions closer in size",
            barrel.span_m(),
            barrel.rise_m()
        ))
        .at(row)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framed_barrels_keep_the_shape_at_a_rise_in_one_to_two() -> Result<(), StructureError> {
        let tiny = 2f64.powi(-400);
        let barrel =
            CulvertBarrel::pipe_arch(1.8542 * tiny, 1.397 * tiny, 0.4572 * tiny, 1.938_528 * tiny)?;
        let circle = CulvertBarrel::circular(0.3 * tiny)?;
        let boxed = CulvertBarrel::rectangular_box(2.0 * tiny, 0.7 * tiny)?;
        for culvert in [barrel, circle, boxed] {
            let framed = framed_barrel(0, &culvert)?;
            assert_eq!(framed.shape(), culvert.shape());
            assert!((1.0..2.0).contains(&framed.rise_m()), "{framed:?}");
            let ratio = framed.rise_m() / culvert.rise_m();
            assert_eq!(ratio, 2f64.powi(ratio.log2().round() as i32));
            assert_eq!(framed.span_m(), culvert.span_m() * ratio);
            let given = framed.given_radius_m().unwrap_or(&[]);
            let expected: Vec<f64> = culvert
                .given_radius_m()
                .unwrap_or(&[])
                .iter()
                .map(|r| r * ratio)
                .collect();
            assert_eq!(given, expected.as_slice());
        }
        // A box whose span would leave the normal range keeps its metre dimensions.
        let wide = CulvertBarrel::rectangular_box(2f64.powi(1000), 2f64.powi(-990))?;
        assert_eq!(framed_barrel(0, &wide)?, wide);
        Ok(())
    }
}
