//! Host enclosures for barrel admission.
//!
//! Each operation rounds its endpoints to nearest, then steps one F64 outward. IEEE arithmetic
//! rounds the exact result to nearest, so each result holds the exact value. An operation on
//! two singletons keeps the exact singleton when an error-free residual proves it exact.
//!
//! The interval kernels of the barrel layout step outward the same way. They keep every exact
//! result this type keeps, under wider guards. Every enclosure here therefore holds the kernel
//! enclosure of the same expression, and a sign this type certifies also holds in the kernel.

/// Magnitudes in `[2^-450, 2^450]`, where error-free residuals cannot underflow or overflow.
const GUARD_MIN: f64 = f64::from_bits((1023 - 450) << 52);
const GUARD_MAX: f64 = f64::from_bits((1023 + 450) << 52);

/// Closed interval `[lower, upper]` holding an exact real value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Enclosure {
    pub(super) lower: f64,
    pub(super) upper: f64,
}

fn guarded(value: f64) -> bool {
    (GUARD_MIN..=GUARD_MAX).contains(&value.abs())
}

impl Enclosure {
    /// The represented value `value`, exactly.
    pub(super) fn exact(value: f64) -> Self {
        Self {
            lower: value,
            upper: value,
        }
    }

    /// Every real number: the result of an operation it cannot bound.
    fn unbounded() -> Self {
        Self {
            lower: f64::NEG_INFINITY,
            upper: f64::INFINITY,
        }
    }

    /// The single value of two guarded nonzero singletons, if both are singletons.
    fn singletons(self, other: Self) -> Option<(f64, f64)> {
        let (a, b) = (self.lower, other.lower);
        let single = a == self.upper && b == other.upper;
        (single && guarded(a) && guarded(b)).then_some((a, b))
    }

    fn outward(lower: f64, upper: f64) -> Self {
        if lower.is_nan() || upper.is_nan() {
            return Self::unbounded();
        }
        Self {
            lower: lower.next_down(),
            upper: upper.next_up(),
        }
    }

    /// `a + b` exactly when TwoSum leaves no residual.
    fn exact_sum(a: f64, b: f64) -> Option<f64> {
        let sum = a + b;
        let virtual_b = sum - a;
        let residual = (a - (sum - virtual_b)) + (b - virtual_b);
        (sum.is_finite() && residual == 0.0).then_some(sum)
    }

    pub(super) fn add(self, other: Self) -> Self {
        if let Some(sum) = self
            .singletons(other)
            .and_then(|(a, b)| Self::exact_sum(a, b))
        {
            return Self::exact(sum);
        }
        Self::outward(self.lower + other.lower, self.upper + other.upper)
    }

    pub(super) fn sub(self, other: Self) -> Self {
        if let Some(sum) = self
            .singletons(other)
            .and_then(|(a, b)| Self::exact_sum(a, -b))
        {
            return Self::exact(sum);
        }
        Self::outward(self.lower - other.upper, self.upper - other.lower)
    }

    pub(super) fn mul(self, other: Self) -> Self {
        if let Some((a, b)) = self.singletons(other) {
            let product = a * b;
            if guarded(product) && a.mul_add(b, -product) == 0.0 {
                return Self::exact(product);
            }
        }
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        let lower = products.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::outward(lower, upper)
    }

    /// Quotient by an enclosure with a positive lower end. Any other divisor bounds nothing.
    pub(super) fn div(self, divisor: Self) -> Self {
        if divisor.lower.is_nan() || divisor.lower <= 0.0 {
            return Self::unbounded();
        }
        if let Some((a, b)) = self.singletons(divisor) {
            let quotient = a / b;
            if guarded(quotient) && quotient.mul_add(b, -a) == 0.0 {
                return Self::exact(quotient);
            }
        }
        let quotients = [
            self.lower / divisor.lower,
            self.lower / divisor.upper,
            self.upper / divisor.lower,
            self.upper / divisor.upper,
        ];
        let lower = quotients.iter().copied().fold(f64::INFINITY, f64::min);
        let upper = quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::outward(lower, upper)
    }

    /// Square root of the nonnegative part. A wholly negative enclosure bounds nothing.
    pub(super) fn sqrt(self) -> Self {
        if self.upper.is_nan() || self.upper < 0.0 {
            return Self::unbounded();
        }
        Self::outward(self.lower.max(0.0).sqrt(), self.upper.sqrt())
    }

    /// The part at or above zero, for a value known to be nonnegative.
    pub(super) fn nonnegative(self) -> Self {
        Self {
            lower: self.lower.max(0.0),
            upper: self.upper.max(0.0),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operations_hold_their_exact_results() {
        let third = Enclosure::exact(1.0).div(Enclosure::exact(3.0));
        // Fused residuals decide 3 lower < 1 < 3 upper exactly.
        assert!(third.lower.mul_add(3.0, -1.0) < 0.0 && third.upper.mul_add(3.0, -1.0) > 0.0);
        let root = Enclosure::exact(2.0).sqrt();
        assert!(root.lower.mul_add(root.lower, -2.0) < 0.0);
        assert!(root.upper.mul_add(root.upper, -2.0) > 0.0);
        let product = Enclosure::exact(-2.0).mul(Enclosure {
            lower: -1.0,
            upper: 3.0,
        });
        assert!(product.lower <= -6.0 && product.upper >= 2.0);
        let sum = Enclosure::exact(1.0).add(Enclosure::exact(f64::EPSILON / 4.0));
        assert!(sum.lower <= 1.0 && sum.upper > 1.0);
    }

    #[test]
    fn exact_singleton_results_stay_singletons() {
        let half = Enclosure::exact(0.5);
        assert_eq!(Enclosure::exact(3.0).mul(half), Enclosure::exact(1.5));
        let below = 0.5_f64.next_down();
        assert_eq!(
            half.sub(Enclosure::exact(below)),
            Enclosure::exact(0.5 - below)
        );
        assert_eq!(
            Enclosure::exact(3.0).div(Enclosure::exact(4.0)),
            Enclosure::exact(0.75)
        );
        // Outside the guard range an exact product still steps outward.
        let tiny = Enclosure::exact(f64::MIN_POSITIVE);
        assert!(tiny.mul(half).lower < f64::MIN_POSITIVE * 0.5);
    }

    #[test]
    fn unbounded_operands_certify_nothing() {
        let zero = Enclosure::exact(0.0);
        let any = Enclosure::exact(1.0).div(zero);
        assert_eq!(any, Enclosure::unbounded());
        assert_eq!(Enclosure::exact(-1.0).sqrt(), Enclosure::unbounded());
        let nan = Enclosure::exact(f64::INFINITY).sub(Enclosure::exact(f64::INFINITY));
        assert_eq!(nan, Enclosure::unbounded());
        assert_eq!(
            Enclosure {
                lower: -1.0,
                upper: 2.0
            }
            .nonnegative(),
            Enclosure {
                lower: 0.0,
                upper: 2.0
            }
        );
    }
}
