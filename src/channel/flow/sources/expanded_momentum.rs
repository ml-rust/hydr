//! Surveyed momentum through withdrawal, transfers, and inflow as two-part expansion updates.

use super::super::conserved_point::ConservedPoint;
use super::super::conserved_update::{
    Arithmetic, ExpansionClient, Parts, bounded, center, exact, magnitude,
};
use super::PreparedLateral;
use numr::dtype::DType;
use numr::ops::ConditionalOps;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::ArithmeticResult;

/// Selected momentum states, the applied withdrawal, and the recorded radii.
pub(super) struct MomentumUpdate<R: Runtime<DType = DType>> {
    pub momentum: ConservedPoint<R>,
    pub withdrawn: ConservedPoint<R>,
    pub transport_radius: Tensor<R>,
    /// Retained-product radius, incoming selection radius, and the receipt relationship.
    pub assembly: [Tensor<R>; 3],
}

/// Retained momentum `(retained_V / V) × M` over both parts of `M` on withdrawal cells.
///
/// Other cells keep `M` with zero radius. The returned radius is discarded by the selection.
fn retained_momentum<R>(
    a: &mut Arithmetic<'_, R>,
    client: &R::Client,
    volume: &ConservedPoint<R>,
    retained: &ConservedPoint<R>,
    momentum: &ConservedPoint<R>,
    active: &Tensor<R>,
) -> ArithmeticResult<(ConservedPoint<R>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let c = client;
    let shape = volume.high.shape();
    let zero = Tensor::zeros(shape, DType::F64, c.device())?;
    let one = Tensor::full_scalar(shape, DType::F64, 1.0, c.device())?;
    // Withdrawal cells hold positive donor water. Other cells divide by one.
    let denominator = exact(
        a,
        &c.where_cond(active, &volume.high, &one)?,
        &c.where_cond(active, &volume.low, &zero)?,
    )?;
    let numerator = exact(a, &retained.high, &retained.low)?;
    let fraction = a.expansion_div_positive(&numerator, &denominator)?;
    let momentum_value = exact(a, &momentum.high, &momentum.low)?;
    let product = a.expansion_mul(&fraction, &momentum_value)?;
    Ok((
        ConservedPoint {
            high: c.where_cond(active, product.high(), &momentum.high)?,
            low: c.where_cond(active, product.low(), &momentum.low)?,
        },
        c.where_cond(active, product.abs_error(), &zero)?,
    ))
}

/// Retain momentum on withdrawal cells, then add the transfers and the incoming momentum.
///
/// The applied withdrawal is `M - retained M`. Its distance from the receipt momentum is
/// recorded as an assembly bound.
pub(super) fn momentum_update<R>(
    a: &mut Arithmetic<'_, R>,
    client: &R::Client,
    [volume, retained_volume]: [&ConservedPoint<R>; 2],
    original: &ConservedPoint<R>,
    active: &Tensor<R>,
    transfer: &Parts<R>,
    prepared: &PreparedLateral<R>,
) -> ArithmeticResult<MomentumUpdate<R>>
where
    R: Runtime<DType = DType>,
    R::Client: ExpansionClient<R>,
{
    let (retained, retained_radius) =
        retained_momentum(a, client, volume, retained_volume, original, active)?;
    let original = exact(a, &original.high, &original.low)?;
    let retained = exact(a, &retained.high, &retained.low)?;
    let withdrawn = a.expansion_sub(&original, &retained)?;
    let receipt = a.expansion_point(&prepared.withdrawal_momentum)?;
    let relationship = a.expansion_sub(&withdrawn, &receipt)?;
    let relationship = magnitude(a, &relationship)?;
    let transfer = bounded(a, transfer)?;
    let transported = a.expansion_add(&retained, &transfer)?;
    let transported_point = exact(a, transported.high(), transported.low())?;
    let incoming = a.expansion_point(&prepared.incoming_momentum)?;
    let accepted = a.expansion_add(&transported_point, &incoming)?;
    Ok(MomentumUpdate {
        momentum: center(&accepted),
        withdrawn: center(&withdrawn),
        transport_radius: transported.abs_error().clone(),
        assembly: [retained_radius, accepted.abs_error().clone(), relationship],
    })
}

#[cfg(test)]
mod tests {
    use super::super::super::conserved_update::expanded;
    use super::super::super::surveyed_update::tests::{Client, TestResult, cpu, exact_sum};
    use super::*;

    fn two_product(a: f64, b: f64) -> [f64; 2] {
        let product = a * b;
        [product, a.mul_add(b, -product)]
    }

    fn retained_momentum_scales_both_parts<R>(client: R::Client) -> TestResult
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let c = &client;
        let tensor = |values: &[f64]| Tensor::<R>::from_slice(values, &[values.len()], c.device());
        let (v_low, m_low) = (2f64.powi(-60), 2f64.powi(-58));
        let point = |high: &[f64], low: &[f64]| -> TestResult<ConservedPoint<R>> {
            Ok(ConservedPoint {
                high: tensor(high)?,
                low: tensor(low)?,
            })
        };
        // Cell 0 withdraws 0.25 m³. Cell 1 has no withdrawal.
        let volume = point(&[1.0, 3.0], &[v_low, 0.0])?;
        let retained = point(&[0.75, 3.0], &[v_low, 0.0])?;
        let momentum = point(&[2.0, 5.0], &[m_low, 2f64.powi(-53)])?;
        let active = tensor(&[1.0, 0.0])?;
        let (selected, radius) = expanded(c, "retained momentum oracle", |a| {
            retained_momentum(a, c, &volume, &retained, &momentum, &active)
        })?;
        let high = selected.high.try_to_vec::<f64>()?;
        let low = selected.low.try_to_vec::<f64>()?;
        let radius = radius.try_to_vec::<f64>()?;
        assert_eq!(high[1].to_bits(), 5f64.to_bits());
        assert_eq!(low[1].to_bits(), 2f64.powi(-53).to_bits());
        assert_eq!(radius[1], 0.0);
        // (h + l) V - retained M, exact. V exceeds one, so this bounds |(h + l) - retained M / V|.
        let error = |h: f64, l: f64| -> f64 {
            let terms = [
                two_product(h, 1.0),
                two_product(h, v_low),
                two_product(l, 1.0),
                two_product(l, v_low),
                two_product(-0.75, 2.0),
                two_product(-0.75, m_low),
                two_product(-v_low, 2.0),
                two_product(-v_low, m_low),
            ];
            exact_sum(terms.into_iter().flatten()).iter().sum()
        };
        let (h, l) = (high[0], low[0]);
        assert_eq!(h + l, h, "the selected center is normalized");
        assert!(error(h, l).abs() <= radius[0], "error={:e}", error(h, l));
        assert!(radius[0] < 2f64.powi(-90), "radius={:e}", radius[0]);
        // Scaling only the high part misses the low-part share by about 2^-61.
        assert!(error(1.5, m_low).abs() > radius[0]);
        Ok(())
    }

    #[test]
    fn cpu_retained_momentum_scales_both_momentum_parts() -> TestResult {
        retained_momentum_scales_both_parts::<numr::runtime::cpu::CpuRuntime>(cpu())
    }

    #[cfg(feature = "cuda")]
    mod cuda {
        use super::*;
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};

        #[test]
        fn cuda_retained_momentum_scales_both_momentum_parts() -> TestResult {
            retained_momentum_scales_both_parts::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
        }
    }
}
