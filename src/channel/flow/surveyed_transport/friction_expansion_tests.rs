use crate::channel::flow::{
    conserved_point::ConservedPoint,
    surveyed_update::tests::{Client, TestResult, cpu, exact_sum, reach_with_roughness, walls},
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::runtime::cpu::CpuRuntime;

fn bits(values: Vec<f64>) -> Vec<u64> {
    values.into_iter().map(f64::to_bits).collect()
}

fn zero_roughness_keeps_both_parts<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let reach = reach_with_roughness::<R>(client, &[10.0, 10.0], &[0.0, 0.0])?;
    let context = &reach.context;
    let volume = context.tensor(&[10.0, 10.0])?;
    let momentum = ConservedPoint {
        high: context.tensor(&[3.0, -3.0])?,
        low: context.tensor(&[2f64.powi(-60), -(2f64.powi(-61))])?,
    };
    let result = context.surveyed_friction_point(&volume, &momentum, 0.1)?;
    let read = |value: &numr::tensor::Tensor<R>| value.try_to_vec::<f64>();
    assert_eq!(bits(read(&result.point.high)?), bits(read(&momentum.high)?));
    assert_eq!(bits(read(&result.point.low)?), bits(read(&momentum.low)?));
    assert_eq!(read(&result.impulse.high)?, vec![0.0, 0.0]);
    assert_eq!(read(&result.impulse.low)?, vec![0.0, 0.0]);
    assert_eq!(read(&result.radius)?, vec![0.0, 0.0]);
    Ok(())
}

fn active_row_records_its_division_radius<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let reach = reach_with_roughness::<R>(client, &[10.0, 10.0], &[0.03, 0.0])?;
    let context = &reach.context;
    let volume = context.tensor(&[10.0, 10.0])?;
    let (hi, lo) = (3.0, 2f64.powi(-60));
    let momentum = ConservedPoint {
        high: context.tensor(&[hi, hi])?,
        low: context.tensor(&[lo, lo])?,
    };
    let drag = context.surveyed_drag(&volume, &momentum.high, 0.1)?;
    let divisor = drag.divisor.ok_or("Active row requires a divisor")?;
    let denominator = divisor.denominator.try_to_vec::<f64>()?;
    assert!(denominator[0] > 1.0);
    assert_eq!(denominator[1], 1.0);
    let result = context.surveyed_friction_point(&volume, &momentum, 0.1)?;
    let high = result.point.high.try_to_vec::<f64>()?;
    let low = result.point.low.try_to_vec::<f64>()?;
    let radius = result.radius.try_to_vec::<f64>()?;
    // The quotient times the divisor, minus the original value, evaluated with a fused product.
    let d = denominator[0];
    let product = high[0] * d;
    let product_error = high[0].mul_add(d, -product);
    let residual = ((product - hi) + product_error) + (low[0] * d - lo);
    assert!(radius[0] > 0.0, "radius={:e}", radius[0]);
    assert!(
        residual.abs() / d <= radius[0],
        "residual={:e}, radius={:e}",
        residual.abs() / d,
        radius[0]
    );
    assert!(radius[0] <= 2f64.powi(-100) * high[0].abs());
    assert!(high[0] < hi);
    // The inactive row keeps both parts bit for bit.
    assert_eq!(high[1].to_bits(), hi.to_bits());
    assert_eq!(low[1].to_bits(), lo.to_bits());
    assert_eq!(radius[1], 0.0);
    let impulse = result.impulse.high.try_to_vec::<f64>()?;
    assert_eq!(impulse[1], 0.0);
    assert!(impulse[0] < 0.0);
    Ok(())
}

fn frictional_steps_conserve_momentum_within_bounds<R>(client: R::Client) -> TestResult
where
    R: Runtime<DType = DType>,
    R::Client: Client<R>,
{
    let mut current = reach_with_roughness::<R>(client, &[30.0, 20.0, 10.0], &[0.03, 0.03, 0.03])?;
    let mut active = false;
    for _ in 0..3 {
        let before = current.snapshot_data()?;
        let start = current.time();
        current = current
            .step(start + 0.01, &walls(start, start + 0.01))?
            .next;
        let after = current.snapshot_data()?;
        let receipt = current.last_step()?.ok_or("Missing frictional receipt")?;
        assert_eq!(receipt.friction_impulse_low_m4_per_s.len(), 3);
        active |= receipt.friction_impulse_m4_per_s.iter().any(|f| *f != 0.0);
        let cells = receipt.friction_impulse_m4_per_s.len();
        let faces = cells;
        let mut terms: Vec<f64> = after
            .longitudinal_momentum_m4_per_s
            .iter()
            .chain(&after.longitudinal_momentum_low_m4_per_s)
            .copied()
            .collect();
        terms.extend(
            before
                .longitudinal_momentum_m4_per_s
                .iter()
                .chain(&before.longitudinal_momentum_low_m4_per_s)
                .map(|part| -part),
        );
        terms.extend([
            -receipt.face_momentum_m4_per_s[0],
            -receipt.face_momentum_low_m4_per_s[0],
            receipt.face_momentum_m4_per_s[faces],
            receipt.face_momentum_low_m4_per_s[faces],
        ]);
        terms.extend(
            receipt.right_pressure_impulse_m4_per_s[..cells]
                .iter()
                .map(|pressure| -pressure),
        );
        terms.extend(receipt.left_pressure_impulse_m4_per_s[1..].iter());
        terms.extend(
            receipt
                .friction_impulse_m4_per_s
                .iter()
                .chain(&receipt.friction_impulse_low_m4_per_s)
                .map(|f| -f),
        );
        let residual: f64 = exact_sum(terms).iter().sum();
        let bound: f64 = receipt.momentum_underflow_bound_m4_per_s.iter().sum();
        assert!(
            residual.abs() <= bound * (1.0 + 4.0 * f64::EPSILON),
            "momentum residual={residual:e}, bound={bound:e}"
        );
    }
    assert!(active, "the fixture must exercise active friction");
    Ok(())
}

#[test]
fn cpu_zero_roughness_keeps_both_momentum_parts() -> TestResult {
    zero_roughness_keeps_both_parts::<CpuRuntime>(cpu())
}

#[test]
fn cpu_active_friction_row_records_its_division_radius() -> TestResult {
    active_row_records_its_division_radius::<CpuRuntime>(cpu())
}

#[test]
fn cpu_frictional_steps_conserve_momentum_within_recorded_bounds() -> TestResult {
    frictional_steps_conserve_momentum_within_bounds::<CpuRuntime>(cpu())
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

    #[test]
    fn cuda_zero_roughness_keeps_both_momentum_parts() -> TestResult {
        zero_roughness_keeps_both_parts::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    #[test]
    fn cuda_active_friction_row_records_its_division_radius() -> TestResult {
        active_row_records_its_division_radius::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }

    #[test]
    fn cuda_frictional_steps_conserve_momentum_within_recorded_bounds() -> TestResult {
        frictional_steps_conserve_momentum_within_bounds::<CudaRuntime>(CudaClient::new(
            CudaDevice::new(0),
        )?)
    }
}
