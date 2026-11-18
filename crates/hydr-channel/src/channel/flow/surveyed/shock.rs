use super::equation_ops::{choose, scalar};
use super::error::{Arithmetic, HydraulicResult};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

pub(crate) fn offset<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let area = shock_area(a, target, original)?;
    offset_from_areas(a, &area, original, secant)
}

pub(crate) fn curve<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let area = shock_area(a, target, original)?;
    let offset = offset_from_areas(a, &area, original, secant)?;
    // The contraction increases with area, so evaluate its represented endpoints independently.
    let lower = contraction_at_area(a, area.lower(), original)?;
    let upper = contraction_at_area(a, area.upper(), original)?;
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let contraction = a.bounds(
        &c.maximum(lower.lower(), zero.lower())?,
        &c.minimum(upper.upper(), one.upper())?,
    )?;
    Ok(a.mul(&contraction, &offset)?)
}

fn shock_area<R>(
    a: &mut Arithmetic<'_, R>,
    target: &Interval<R>,
    original: &Tensor<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    // Rarefaction-only rows retain a benign hypothetical shock at the original area.
    Ok(a.bounds(
        &c.maximum(target.lower(), original)?,
        &c.maximum(target.upper(), original)?,
    )?)
}

fn offset_from_areas<R>(
    a: &mut Arithmetic<'_, R>,
    area: &Interval<R>,
    original: &Tensor<R>,
    secant: &Interval<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let multiply_first = c.mul(
        &c.lt(original, one.lower())?,
        &c.ge(area.upper(), one.upper())?,
    )?;
    let divide_first = c.eq(&multiply_first, zero.lower())?;
    let original = a.point(original)?;
    let original_root = a.sqrt_nonnegative(&original)?;
    let area_root = a.sqrt_nonnegative(area)?;
    let secant_root = a.sqrt_nonnegative(secant)?;

    // Mask every unused operand before eager interval multiplication or division.
    let product_secant = choose(a, &multiply_first, &secant_root, &one)?;
    let product_area = choose(a, &multiply_first, &area_root, &one)?;
    let product_original = choose(a, &multiply_first, &original_root, &one)?;
    let product = a.mul(&product_secant, &product_area)?;
    let product_first = a.div_positive_interval(&product, &product_original)?;

    let quotient_secant = choose(a, &divide_first, &secant_root, &one)?;
    let quotient_area = choose(a, &divide_first, &area_root, &one)?;
    let quotient_original = choose(a, &divide_first, &original_root, &one)?;
    let quotient = a.div_positive_interval(&quotient_area, &quotient_original)?;
    let quotient_first = a.mul(&quotient_secant, &quotient)?;
    choose(a, &multiply_first, &product_first, &quotient_first)
}

fn contraction_at_area<R>(
    a: &mut Arithmetic<'_, R>,
    area: &Tensor<R>,
    original: &Tensor<R>,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let close = c.ge(original, &c.mul_scalar(area, 0.5)?)?;
    let distant = c.eq(&close, zero.lower())?;
    let equal = c.eq(area, original)?;
    let area = a.point(area)?;
    let original = a.point(original)?;

    let close_area = choose(a, &close, &area, &one)?;
    let close_original = choose(a, &close, &original, &one)?;
    // Sterbenz subtraction is exact here, including gradual underflow and equal areas.
    let difference = a.point(&c.sub(close_area.lower(), close_original.lower())?)?;
    let close_ratio = a.div_positive_interval(&difference, &close_area)?;

    let distant_area = choose(a, &distant, &area, &one)?;
    let distant_original = choose(a, &distant, &original, &one)?;
    let quotient = a.div_positive_interval(&distant_original, &distant_area)?;
    let distant_ratio = a.sub(&one, &quotient)?;
    let ratio = choose(a, &close, &close_ratio, &distant_ratio)?;
    choose(a, &equal, &zero, &ratio)
}

#[cfg(test)]
mod tests {
    use super::super::error::{PrismaticRiemannError, checked, into_public};
    use super::super::potential_difference_test_support::values;
    use super::super::shock;
    use super::super::shock_case_data::CASES;
    use super::*;
    use numr::ops::{HypotOps, MidpointOps};

    fn contains<R>(
        value: &Interval<R>,
        expected: &[[u64; 2]],
        labels: &[&str],
    ) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
    {
        let lower = value.lower().try_to_vec::<f64>()?;
        let upper = value.upper().try_to_vec::<f64>()?;
        assert_eq!(lower.len(), expected.len());
        assert_eq!(upper.len(), expected.len());
        assert_eq!(labels.len(), expected.len());
        for (((&lower, &upper), reference), label) in
            lower.iter().zip(&upper).zip(expected).zip(labels)
        {
            let reference = reference.map(f64::from_bits);
            assert!(
                lower.is_finite() && upper.is_finite() && lower <= upper,
                "{label}: nonfinite or unordered [{lower:e}, {upper:e}]"
            );
            assert!(
                lower <= reference[0] && upper >= reference[1],
                "{label}: actual=[{lower:e}, {upper:e}], reference={reference:?}"
            );
        }
        Ok(())
    }

    fn references<R>(client: &R::Client, strided: bool) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        assert_eq!(CASES.len(), 12);
        let original: Vec<_> = CASES
            .iter()
            .map(|case| f64::from_bits(case.original))
            .collect();
        let lower: Vec<_> = CASES
            .iter()
            .map(|case| f64::from_bits(case.target[0]))
            .collect();
        let upper: Vec<_> = CASES
            .iter()
            .map(|case| f64::from_bits(case.target[1]))
            .collect();
        let secant: Vec<_> = CASES
            .iter()
            .map(|case| f64::from_bits(case.secant))
            .collect();
        let original = values::<R, _>(client, &original, strided)?;
        let lower = values::<R, _>(client, &lower, strided)?;
        let upper = values::<R, _>(client, &upper, strided)?;
        let secant = values::<R, _>(client, &secant, strided)?;
        let (speed, curve) = checked(client, "independent mixed shock algebra references", |a| {
            let target = a.bounds(&lower, &upper)?;
            let secant = a.point(&secant)?;
            Ok((
                shock::offset(a, &target, &original, &secant)?,
                shock::curve(a, &target, &original, &secant)?,
            ))
        })?;
        let labels: Vec<_> = CASES.iter().map(|case| case.id).collect();
        contains(
            &speed,
            &CASES.iter().map(|case| case.speed).collect::<Vec<_>>(),
            &labels,
        )?;
        contains(
            &curve,
            &CASES.iter().map(|case| case.curve).collect::<Vec<_>>(),
            &labels,
        )?;
        assert_eq!(curve.lower().try_to_vec::<f64>()?[0], 0.0);
        assert_eq!(curve.upper().try_to_vec::<f64>()?[0], 0.0);
        Ok(())
    }

    fn adjacent_areas<R>(client: &R::Client, strided: bool) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        for span in [false, true] {
            let mut original = Vec::new();
            let mut lower = Vec::new();
            let mut upper = Vec::new();
            let mut expected = Vec::new();
            for exponent in [-600, 0, 600] {
                let area = 2.0_f64.powi(exponent);
                let low = f64::from_bits(area.to_bits() + 1);
                let high = f64::from_bits(area.to_bits() + if span { 2 } else { 1 });
                original.push(area);
                lower.push(low);
                upper.push(high);
                // With secant=A, the exact curve is (A-Ai)/sqrt(Ai).
                let difference = 2.0_f64.powi(exponent / 2 - 52);
                expected.push([
                    difference.to_bits(),
                    (difference * if span { 2.0 } else { 1.0 }).to_bits(),
                ]);
            }
            let original = values::<R, _>(client, &original, strided)?;
            let lower = values::<R, _>(client, &lower, strided)?;
            let upper = values::<R, _>(client, &upper, strided)?;
            let curve = checked(client, "exact adjacent-area shock differences", |a| {
                let target = a.bounds(&lower, &upper)?;
                let secant = a.bounds(&lower, &upper)?;
                shock::curve(a, &target, &original, &secant)
            })?;
            contains(
                &curve,
                &expected,
                &["tiny adjacent", "unit adjacent", "large adjacent"],
            )?;
            assert!(
                curve
                    .lower()
                    .try_to_vec::<f64>()?
                    .iter()
                    .all(|&value| value > 0.0)
            );
        }
        Ok(())
    }

    fn subnormal_half_ties<R>(client: &R::Client, strided: bool) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let original = values::<R, _>(client, &[f64::from_bits(1), f64::from_bits(2)], strided)?;
        let target = values::<R, _>(client, &[f64::from_bits(3), f64::from_bits(5)], strided)?;
        let secant = values::<R, _>(client, &[f64::from_bits(3), f64::from_bits(10)], strided)?;
        let (speed, curve) = checked(
            client,
            "exact subnormal rounded-half shock identities",
            |a| {
                let target = a.point(&target)?;
                let secant = a.point(&secant)?;
                Ok((
                    shock::offset(a, &target, &original, &secant)?,
                    shock::curve(a, &target, &original, &secant)?,
                ))
            },
        )?;
        // MIN=2^-1074. The squared speeds are exactly 9*MIN and 25*MIN.
        let root_minimum = f64::from_bits(0x1e60000000000000);
        let labels = ["subnormal half rounds up", "subnormal half rounds down"];
        contains(
            &speed,
            &[
                [(3.0 * root_minimum).to_bits(); 2],
                [(5.0 * root_minimum).to_bits(); 2],
            ],
            &labels,
        )?;
        // Differences remain exact integer multiples of MIN despite rounded A/2 ties.
        contains(
            &curve,
            &[
                [(2.0 * root_minimum).to_bits(); 2],
                [(3.0 * root_minimum).to_bits(); 2],
            ],
            &labels,
        )?;
        assert!(
            curve
                .lower()
                .try_to_vec::<f64>()?
                .iter()
                .all(|&value| value > 0.0)
        );
        Ok(())
    }

    fn complete_span<R>(client: &R::Client, strided: bool) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        let minimum = f64::from_bits(1);
        let original = values::<R, _>(client, &[minimum, 1.0], strided)?;
        let lower = values::<R, _>(client, &[minimum, 0.0], strided)?;
        let upper = values::<R, _>(client, &[f64::MAX, 0.5], strided)?;
        let secant = values::<R, _>(client, &[minimum, 1.0], strided)?;
        let physical_rarefaction = client.lt(&upper, &original)?.try_to_vec::<f64>()?;
        let read_input = |source: &Tensor<R>| -> HydraulicResult<Vec<f64>> {
            let copy = Tensor::empty(source.shape(), source.dtype(), client.device())?;
            client.copy_into(&copy, source)?;
            Ok(copy.try_to_vec::<f64>()?)
        };
        let original_lower = read_input(&lower)?;
        let original_upper = read_input(&upper)?;
        let (speed, curve) = checked(
            client,
            "complete shock span and hypothetical rarefaction",
            |a| {
                let target = a.bounds(&lower, &upper)?;
                let secant = a.point(&secant)?;
                Ok((
                    shock::offset(a, &target, &original, &secant)?,
                    shock::curve(a, &target, &original, &secant)?,
                ))
            },
        )?;
        contains(
            &speed,
            &[
                [0x1e60000000000000, 0x5ff0000000000000],
                [1.0_f64.to_bits(); 2],
            ],
            &["minimum-to-maximum speed", "clamped rarefaction speed"],
        )?;
        contains(
            &curve,
            &[[0, 0x5ff0000000000000], [0, 0]],
            &["minimum-to-maximum curve", "clamped rarefaction curve"],
        )?;
        assert_eq!(curve.lower().try_to_vec::<f64>()?[1], 0.0);
        assert_eq!(curve.upper().try_to_vec::<f64>()?[1], 0.0);
        assert_eq!(physical_rarefaction, [0.0, 1.0]);
        assert_eq!(
            client.lt(&upper, &original)?.try_to_vec::<f64>()?,
            physical_rarefaction
        );
        assert_eq!(read_input(&lower)?, original_lower);
        assert_eq!(read_input(&upper)?, original_upper);
        Ok(())
    }

    fn algebra<R>(client: R::Client) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>,
    {
        for strided in [false, true] {
            references::<R>(&client, strided)?;
            adjacent_areas::<R>(&client, strided)?;
            subnormal_half_ties::<R>(&client, strided)?;
            complete_span::<R>(&client, strided)?;
        }
        Ok(())
    }

    #[test]
    fn shock_algebra_encloses_independent_references_and_exact_differences_cpu()
    -> Result<(), PrismaticRiemannError> {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        algebra::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn shock_algebra_encloses_independent_references_and_exact_differences_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        algebra::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
            .map_err(|error| into_public(error, Default::default()))?;
        Ok(())
    }
}
