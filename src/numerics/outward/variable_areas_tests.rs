//! Variable-area correction products share the scalar path's outward accounting.

use numr::runtime::cpu::CpuRuntime;

use super::*;
use crate::numerics::outward::add_adjustment_charge;
use numr::runtime::cpu::{CpuClient, CpuDevice, ParallelismConfig};

fn setup() -> (CpuDevice, CpuClient) {
    let device = CpuDevice::new();
    let client =
        CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
    (device, client)
}

fn tensor(
    device: &CpuDevice,
    values: &[f64],
) -> Result<numr::tensor::Tensor<CpuRuntime>, OutwardError> {
    Ok(numr::tensor::Tensor::<CpuRuntime>::from_slice(
        values,
        &[values.len()],
        device,
    )?)
}

#[test]
fn equal_areas_match_scalar_bits_at_simd_and_tail_lengths() -> Result<(), OutwardError> {
    let (device, client) = setup();
    for count in [1, 31, 32, 33, 64, 65] {
        let values: Vec<_> = (0..count)
            .map(|i| match i % 5 {
                0 => 0.0,
                1 => -0.0,
                2 => 2f64.powi(-70),
                3 => 3.0 * 2f64.powi(-80),
                _ => 2f64.powi(-400),
            })
            .collect();
        let depths = tensor(&device, &values)?;
        let reduction = AdjustmentReduction::new(count)?;
        for area in [0.3, 0.5, 4.0, 2f64.powi(300)] {
            let areas = tensor(&device, &vec![area; count])?;
            let scalar = reduction.totals(&client, &depths, area)?;
            let variable = reduction.totals_with_areas(&client, &depths, &areas)?;
            assert_eq!(scalar.0.to_bits(), variable.0.to_bits());
            assert_eq!(scalar.1.to_bits(), variable.1.to_bits());
        }
    }
    Ok(())
}

#[test]
fn unequal_power_of_two_products_have_known_reported_sum() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let depths = tensor(&device, &[1.0, 2.0, 4.0, 8.0])?;
    let areas = tensor(&device, &[8.0, 4.0, 2.0, 1.0])?;
    let (reported, charge) =
        AdjustmentReduction::new(4)?.totals_with_areas(&client, &depths, &areas)?;
    assert_eq!(reported, 32.0);
    assert!(charge > reported);
    let next = add_adjustment_charge(1.0, charge)?;
    assert!(next > 1.0 + reported);
    Ok(())
}

#[test]
fn empty_and_zero_corrections_preserve_zero_charge() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let empty = tensor(&device, &[])?;
    let reduction = AdjustmentReduction::new(0)?;
    assert_eq!(reduction.totals(&client, &empty, 1.0)?, (0.0, 0.0));
    assert_eq!(
        reduction.totals_with_areas(&client, &empty, &empty)?,
        (0.0, 0.0)
    );
    let depths = tensor(&device, &[0.0, -0.0, 0.0])?;
    let areas = tensor(&device, &[0.5, 2.0, 2f64.powi(400)])?;
    let totals = AdjustmentReduction::new(3)?.totals_with_areas(&client, &depths, &areas)?;
    assert_eq!(totals.0.to_bits(), 0.0f64.to_bits());
    assert_eq!(totals.1.to_bits(), 0.0f64.to_bits());
    assert_eq!(
        add_adjustment_charge(-0.0, totals.1)?.to_bits(),
        (-0.0f64).to_bits()
    );
    Ok(())
}

#[test]
fn positive_product_underflow_identifies_the_rejected_lane() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let depths = tensor(&device, &[1.0, f64::from_bits(1), 0.0])?;
    let areas = tensor(&device, &[1.0, 0.5, 1.0])?;
    let result = AdjustmentReduction::new(3)?.totals_with_areas(&client, &depths, &areas);
    let error = result
        .err()
        .ok_or_else(|| OutwardError::numerical("underflow unexpectedly returned totals"))?;
    assert_eq!(error.cell(), Some(1));
    assert!(error.is_retryable());
    Ok(())
}

#[test]
fn overflowing_product_and_outward_bound_reject_with_lane_identity() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let depths = tensor(&device, &[0.0, f64::MAX])?;
    for area in [1.0, 2.0] {
        let areas = tensor(&device, &[1.0, area])?;
        let error = AdjustmentReduction::new(2)?
            .totals_with_areas(&client, &depths, &areas)
            .err()
            .ok_or_else(|| OutwardError::numerical("overflow unexpectedly returned totals"))?;
        assert_eq!(error.cell(), Some(1));
        assert!(error.is_retryable());
    }
    Ok(())
}

#[test]
fn invalid_area_values_are_typed_static_lane_errors() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let depths = tensor(&device, &[0.0, 0.0, 0.0])?;
    for area in [0.0, -0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let areas = tensor(&device, &[1.0, 2.0, area])?;
        let error = AdjustmentReduction::new(3)?
            .totals_with_areas(&client, &depths, &areas)
            .err()
            .ok_or_else(|| OutwardError::numerical("invalid area unexpectedly returned totals"))?;
        assert_eq!(error.cell(), Some(2));
        assert!(!error.is_retryable());
    }
    Ok(())
}

#[test]
fn mismatched_flat_shape_and_dtype_reject_without_lane_guessing() -> Result<(), OutwardError> {
    let (device, client) = setup();
    let depths = tensor(&device, &[1.0, 2.0])?;
    let correct = tensor(&device, &[1.0, 1.0])?;
    let wrong_count = tensor(&device, &[1.0])?;
    let matrix = correct.reshape(&[1, 2])?;
    let f32_areas = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32, 1.0], &[2], &device)?;
    let int_areas = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[1i64, 1], &[2], &device)?;
    for areas in [&wrong_count, &matrix, &f32_areas, &int_areas] {
        let error = AdjustmentReduction::new(2)?
            .totals_with_areas(&client, &depths, areas)
            .err()
            .ok_or_else(|| {
                OutwardError::numerical("invalid metadata unexpectedly returned totals")
            })?;
        assert_eq!(error.cell(), None);
        assert!(!error.is_retryable());
    }
    let f32_depths = numr::tensor::Tensor::<CpuRuntime>::from_slice(&[1.0f32, 2.0], &[2], &device)?;
    assert!(
        AdjustmentReduction::new(2)?
            .totals_with_areas(&client, &f32_depths, &correct)
            .is_err()
    );
    assert!(
        AdjustmentReduction::new(2)?
            .totals(&client, &f32_depths, 1.0)
            .is_err()
    );
    Ok(())
}
