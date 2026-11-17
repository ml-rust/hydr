use super::{ChannelError, context::Context};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Two-part conserved cell values on the caller's device.
///
/// `(high, low)` is normalized: `two_sum(high, low) == (high, low)`. The radius is zero.
#[derive(Clone, Debug)]
pub(super) struct ConservedPoint<R: Runtime<DType = DType>> {
    pub high: Tensor<R>,
    pub low: Tensor<R>,
}

impl<R: Runtime<DType = DType>> ConservedPoint<R> {
    /// Point with a `+0.0` low part on the high part's device.
    pub fn from_high(high: &Tensor<R>) -> Result<Self, ChannelError> {
        Ok(Self {
            high: high.clone(),
            low: Tensor::zeros(high.shape(), high.dtype(), high.device())?,
        })
    }

    /// Upload host parts after checking the low part against the geometry and the high part.
    pub fn from_parts(
        context: &Context<R>,
        field: &str,
        high: &[f64],
        low: &[f64],
    ) -> Result<Self, ChannelError> {
        check_parts(field, high, low, context.surveyed())?;
        Ok(Self {
            high: context.tensor(high)?,
            low: context.tensor(low)?,
        })
    }
}

/// Reject a low part that is not normalized, or nonzero on non-surveyed geometry.
pub(super) fn check_parts(
    field: &str,
    high: &[f64],
    low: &[f64],
    surveyed: bool,
) -> Result<(), ChannelError> {
    if high.len() != low.len() {
        return Err(ChannelError::input(format!(
            "{field} high count={}, low count={}: supply one low value per cell",
            high.len(),
            low.len()
        )));
    }
    for (cell, (&h, &l)) in high.iter().zip(low).enumerate() {
        if !surveyed && l.to_bits() != 0 {
            return Err(ChannelError::input(format!(
                "{field} low={l}: use +0.0 on rectangular or triangular geometry"
            ))
            .at(cell));
        }
        // A non-finite high part is rejected by the state field check.
        if h.is_finite() && !normalized(h, l) {
            return Err(ChannelError::input(format!(
                "{field} high={h}, low={l}: supply finite parts with two_sum(high, low) == (high, low)"
            ))
            .at(cell));
        }
    }
    Ok(())
}

fn normalized(high: f64, low: f64) -> bool {
    if !high.is_finite() || !low.is_finite() {
        return false;
    }
    let sum = high + low;
    let virtual_low = sum - high;
    let error = (high - (sum - virtual_low)) + (low - virtual_low);
    sum == high && error == low
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_low_is_normalized() {
        assert!(normalized(2.5, 0.0));
        assert!(normalized(-0.0, 0.0));
    }

    #[test]
    fn low_below_half_ulp_is_normalized() {
        assert!(normalized(1.0, 1e-20));
        assert!(!normalized(1.0, 0.5));
        assert!(!normalized(1.0, f64::NAN));
        assert!(!normalized(f64::INFINITY, 0.0));
    }

    #[test]
    fn rectangular_geometry_rejects_a_nonzero_low() {
        let error = check_parts("volume", &[1.0, 1.0], &[0.0, 1e-20], false);
        assert_eq!(error.as_ref().err().and_then(ChannelError::cell), Some(1));
        assert!(check_parts("volume", &[1.0, 1.0], &[0.0, 1e-20], true).is_ok());
        let error = check_parts("volume", &[1.0, 1.0], &[0.0, -0.0], false);
        assert_eq!(error.as_ref().err().and_then(ChannelError::cell), Some(1));
    }

    #[test]
    fn unnormalized_low_is_rejected_on_every_geometry() {
        assert!(check_parts("volume", &[1.0], &[0.5], true).is_err());
        assert!(check_parts("volume", &[1.0], &[0.5], false).is_err());
    }
}
