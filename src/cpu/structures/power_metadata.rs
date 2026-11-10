//! Static supplied IEEE power-of-two classification without numerical state advancement.

pub(crate) fn power_of_two(value: f64) -> bool {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        fraction != 0 && (fraction & (fraction - 1)) == 0
    } else {
        fraction == 0
    }
}
