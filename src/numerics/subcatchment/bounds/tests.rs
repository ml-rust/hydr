use super::*;
use crate::runoff::{ReservoirConfig, ReservoirElement, ReservoirParameters};

fn context(count: usize) -> Result<CpuReservoirContext, ReservoirError> {
    let parameters = ReservoirParameters::from_si(1.0, 1.0, 0.0, 0.0, 1.0, 0.0)?;
    CpuReservoirContext::new(
        (0..count)
            .map(|i| ReservoirElement {
                id: format!("bound-{i}"),
                parameters,
            })
            .collect(),
        ReservoirConfig::default(),
    )
}

#[test]
fn downward_budgets_handle_subnormals_and_exact_absolute_allowance() -> Result<(), ReservoirError> {
    let tiny = f64::from_bits(1);
    assert_eq!(permitted_budget(tiny, 0.0, f64::MAX)?, tiny);
    assert_eq!(permitted_budget(tiny, 1.0, 0.0)?, tiny);
    assert_eq!(permitted_budget(0.0, 0.5, tiny)?, 0.0);
    assert_eq!(permitted_budget(tiny, 0.5, tiny)?, tiny);
    let budget = permitted_budget(0.0, 0.5, 1.0)?;
    assert!(budget > 0.0 && budget < 0.5);
    assert!(permitted_budget(1.0, f64::MAX, 2.0).is_err());
    Ok(())
}

#[test]
fn growing_physical_scale_never_erodes_accepted_lower_bound() -> Result<(), ReservoirError> {
    let tiny = f64::from_bits(1);
    assert_eq!(add_scale_lower(0.0, tiny)?, tiny);
    assert_eq!(add_scale_lower(1.0, tiny)?, 1.0);
    assert_eq!(add_scale_lower(-0.0, 0.0)?.to_bits(), (-0.0f64).to_bits());
    let mut lower = 1.0;
    for increment in [tiny, 2f64.powi(-70), 1.0, 4.0, tiny] {
        let next = add_scale_lower(lower, increment)?;
        assert!(next >= lower && next >= increment);
        lower = next;
    }
    assert!(add_scale_lower(f64::MAX, f64::MAX).is_err());
    Ok(())
}

#[test]
fn positive_charges_consume_budget_when_ordinary_addition_stagnates() -> Result<(), ReservoirError>
{
    let tiny = f64::from_bits(1);
    assert_eq!(1.0 + tiny, 1.0);
    let mut charge = 1.0;
    let permitted = 1.0 + 16.0 * f64::EPSILON;
    for _ in 0..32 {
        charge = add_charge(charge, tiny)?;
    }
    assert!(charge > permitted);
    assert_eq!(add_charge(-0.0, 0.0)?.to_bits(), (-0.0f64).to_bits());
    Ok(())
}

#[test]
fn reduction_lower_bound_never_exceeds_exact_power_of_two_sum() -> Result<(), ReservoirError> {
    for count in [1, 31, 32, 33, 64, 65] {
        let ctx = context(count)?;
        // Integer multiples remain exactly representable, independent of reduction order.
        let values: Vec<f64> = (0..count)
            .map(|i| (i + 1) as f64 * 2f64.powi(-20))
            .collect();
        let exact = (count * (count + 1) / 2) as f64 * 2f64.powi(-20);
        let tensor = ctx.tensor(&values)?;
        let lower = sum_lower(&ctx, &tensor)?;
        assert!(lower > 0.0 && lower <= exact);
        assert!(sum_upper(&ctx, &tensor)? >= exact);
        assert_eq!(sum_lower(&ctx, &ctx.zero)?, 0.0);
        assert_eq!(sum_upper(&ctx, &ctx.zero)?, 0.0);
    }
    Ok(())
}

#[test]
fn positive_uncertainty_survives_a_zero_scale_increment() -> Result<(), ReservoirError> {
    let ctx = context(2)?;
    let tiny = ctx.tensor(&[f64::from_bits(1), 0.0])?;
    assert_eq!(sum_lower(&ctx, &tiny)?, 0.0);
    let charge = sum_upper(&ctx, &tiny)?;
    assert!(charge > 0.0);
    assert!(add_charge(0.0, charge)? > permitted_budget(0.0, 1.0, 0.0)?);
    Ok(())
}
