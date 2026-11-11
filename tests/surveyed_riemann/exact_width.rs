// Exact dyadic sums check the original represented allowance without rounding slack.
const LIMBS: usize = 68;
const BASE: i32 = -2148;

fn operand(value: f64) -> (u64, i32) {
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 52) & 2047) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        (fraction, -1074)
    } else {
        (fraction | (1u64 << 52), exponent - 1075)
    }
}

fn add(target: &mut [u64; LIMBS], coefficient: u128, exponent: i32) -> super::support::Result {
    let shift = usize::try_from(exponent - BASE)
        .map_err(|error| format!("Finite dyadic exponent={exponent}, base={BASE}: {error}"))?;
    for bit in 0..128 {
        if coefficient & (1u128 << bit) == 0 {
            continue;
        }
        let position = shift + bit;
        let mut index = position / 64;
        let mut carry = 1u64 << (position % 64);
        loop {
            let (sum, overflow) = target[index].overflowing_add(carry);
            target[index] = sum;
            if !overflow {
                break;
            }
            index += 1;
            carry = 1;
        }
    }
    Ok(())
}

pub(super) fn accepted(
    lower: f64,
    upper: f64,
    atol: f64,
    rtol: f64,
) -> super::support::Result<bool> {
    assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
    assert!(atol.is_finite() && atol >= 0.0 && rtol.is_finite() && rtol >= 0.0);
    let mut positive = [0u64; LIMBS];
    let mut negative = [0u64; LIMBS];
    let (a, ae) = operand(atol);
    let (r, re) = operand(rtol);
    let (scale, se) = operand(lower.abs().max(upper.abs()));
    add(&mut positive, u128::from(a), ae)?;
    add(&mut positive, u128::from(r) * u128::from(scale), re + se)?;
    for (value, subtract) in [(lower, false), (upper, true)] {
        let (mantissa, exponent) = operand(value);
        let target = if value.is_sign_negative() ^ subtract {
            &mut negative
        } else {
            &mut positive
        };
        add(target, u128::from(mantissa), exponent)?;
    }
    Ok(positive.iter().rev().cmp(negative.iter().rev()).is_ge())
}

#[test]
fn exact_allowance_retains_sub_ulp_excess() -> super::support::Result {
    assert!(accepted(1.0, 1.0, 0.0, 0.0)?);
    assert!(accepted(-1.0, 1.0, 2.0, 0.0)?);
    assert!(!accepted(-f64::from_bits(1), 1.0, 1.0, 0.0)?);
    assert!(accepted(0.0, f64::from_bits(1), f64::from_bits(1), 0.0)?);
    assert!(!accepted(0.0, f64::from_bits(2), f64::from_bits(1), 0.0)?);
    assert!(accepted(-f64::MAX, f64::MAX, f64::MAX, 1.0)?);
    Ok(())
}
