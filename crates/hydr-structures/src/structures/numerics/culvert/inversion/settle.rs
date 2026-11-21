//! Stage 2 of the outlet root search: two fine marches narrow a bracket to the noise floor.
//!
//! One fine march at both bracket ends gives a secant root `x` and the residual noise. One more
//! fine march at `x - d` and `x + d` then certifies the root between them, where `d` covers a
//! few noise widths over the secant slope and the secant's curvature miss.

use super::Selected;
use super::lanes::Lanes;
use super::outlet_root::gap_at;
use crate::structures::numerics::culvert::outlet::Accuracy;
use crate::structures::tensor::culvert::{Bounds, TensorCulvertLaw};
use crate::structures::{CulvertClient, StructureError};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ScalarOps, ShapeOps, UnaryOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Noise widths over the secant slope on each side of the secant root.
const NOISE_MARGIN: f64 = 4.0;

/// Relative floor of the stage 2 offset, `2^-48` of the secant root.
const OFFSET_FLOOR: f64 = 1.0 / 281_474_976_710_656.0;

/// A stage 2 bracket, with masks of the lanes whose secant probes straddle the root and of the
/// lanes whose bracket ends certify their signs.
pub(super) struct Settled<R: Runtime<DType = DType>> {
    pub(super) bounds: Bounds<R>,
    pub(super) straddled: Tensor<R>,
    /// The fine residual is certified at most zero at the lower end and at least zero at the
    /// upper end, so the root lies in the bracket.
    pub(super) verified: Tensor<R>,
}

/// Fine residuals at the bracket ends, then at the secant root plus and minus the offset.
///
/// Returns the bracket narrowed by every certified sign. Lanes outside `wanted` keep `bracket`.
pub(super) fn settle<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    wanted: &Tensor<R>,
    bracket: &Bounds<R>,
) -> Result<Settled<R>, StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let zero = &barrel.zero;
    // Lanes outside `wanted` march dry at zero discharge.
    let low = c.where_cond(wanted, &bracket[0], zero)?;
    let high = c.where_cond(wanted, &bracket[1], zero)?;
    let [at_low, at_high] = pair(law, rows, lanes, &low, &high)?;
    let verified = c.mul(
        wanted,
        &c.mul(&c.le(&at_low[1], zero)?, &c.ge(&at_high[0], zero)?)?,
    )?;
    let middle = |value: &Bounds<R>| -> numr::error::Result<Tensor<R>> {
        c.mul_scalar(&c.add(&value[0], &value[1])?, 0.5)
    };
    let (r_low, r_high) = (middle(&at_low)?, middle(&at_high)?);
    let width = c.sub(&high, &low)?;
    let rise = c.sub(&r_high, &r_low)?;
    let slope = c.div(&rise, &width)?;
    let usable = c.mul(&c.gt(&rise, zero)?, &c.gt(&width, zero)?)?;
    let step = c.div(
        &c.mul(&r_low, &width)?,
        &c.where_cond(&usable, &rise, &barrel.one)?,
    )?;
    let secant = c.sub(&low, &step)?;
    let secant = c.minimum(&c.maximum(&secant, &low)?, &high)?;
    let noise = c.maximum(
        &c.sub(&at_low[1], &at_low[0])?,
        &c.sub(&at_high[1], &at_high[0])?,
    )?;
    let offset = c.div(
        &c.mul_scalar(&noise, NOISE_MARGIN)?,
        &c.where_cond(&usable, &slope, &barrel.one)?,
    )?;
    // `HW_o` grows about as `Q^2`, so the secant misses the root by about `W^2 / (8 Q)` over a
    // bracket of width `W`. The offset covers four times that.
    let magnitude = c.maximum(&c.abs(&secant)?, &c.mul_scalar(&width, 0.5)?)?;
    let curvature = c.div(
        &c.mul_scalar(&c.mul(&width, &width)?, 0.5)?,
        &c.where_cond(&c.gt(&magnitude, zero)?, &magnitude, &barrel.one)?,
    )?;
    let offset = c.add(&offset, &curvature)?;
    let offset = c.maximum(&offset, &c.mul_scalar(&c.abs(&secant)?, OFFSET_FLOOR)?)?;
    // Without a usable secant, the probes split the bracket in thirds.
    let third = c.mul_scalar(&width, 1.0 / 3.0)?;
    let left = c.where_cond(&usable, &c.sub(&secant, &offset)?, &c.add(&low, &third)?)?;
    let right = c.where_cond(&usable, &c.add(&secant, &offset)?, &c.sub(&high, &third)?)?;
    let left = c.minimum(&c.maximum(&left, &low)?, &high)?;
    let right = c.minimum(&c.maximum(&right, &left)?, &high)?;
    let [at_left, at_right] = pair(law, rows, lanes, &left, &right)?;
    // A residual certified at most zero puts the root at or above its point, and one certified
    // at least zero puts it at or below.
    let mut lower = low.clone();
    let mut upper = high.clone();
    for (point, value) in [(&left, &at_left), (&right, &at_right)] {
        let below = c.le(&value[1], zero)?;
        let above = c.ge(&value[0], zero)?;
        lower = c.where_cond(&below, &c.maximum(&lower, point)?, &lower)?;
        upper = c.where_cond(&above, &c.minimum(&upper, point)?, &upper)?;
    }
    let ordered = c.mul(wanted, &c.le(&lower, &upper)?)?;
    // The secant probes straddle the root: the bracket is as narrow as the noise allows.
    let straddled = c.mul(
        &c.mul(&c.le(&at_left[1], zero)?, &c.ge(&at_right[0], zero)?)?,
        &usable,
    )?;
    Ok(Settled {
        bounds: [
            c.where_cond(&ordered, &lower, &bracket[0])?,
            c.where_cond(&ordered, &upper, &bracket[1])?,
        ],
        straddled: c.mul(&ordered, &straddled)?,
        verified,
    })
}

/// Fine residuals of every lane at `first` and at `second`, in one march over both.
fn pair<R>(
    law: &TensorCulvertLaw<R>,
    rows: &Selected<R>,
    lanes: &Lanes<R>,
    first: &Tensor<R>,
    second: &Tensor<R>,
) -> Result<[Bounds<R>; 2], StructureError>
where
    R: Runtime<DType = DType>,
    R::Client: CulvertClient<R>,
{
    let barrel = &rows.barrel;
    let c = &barrel.client;
    let m = barrel.width();
    let lane: Vec<i64> = (0..2 * m)
        .map(|index| i64::try_from(index % m).unwrap_or(i64::MAX))
        .collect();
    let owner = Tensor::<R>::from_slice(&lane, &[2 * m], &barrel.device)?;
    let point = c.cat(&[first, second], 0)?;
    let [lower, upper] = gap_at(law, rows, lanes, &owner, &point, Accuracy::Fine)?;
    let half = |value: &Tensor<R>, index: usize| value.narrow(0, index * m, m);
    Ok([
        [half(&lower, 0)?, half(&upper, 0)?],
        [half(&lower, 1)?, half(&upper, 1)?],
    ])
}
