//! Batched bisection of the star-region depth over reference-rooted wave curves.

use crate::contract::{EngineError, SolverConfig};
use crate::numerics::tensor_types::{check_finite, constant};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, UnaryOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::optimize::OptimizeError;
use solvr::optimize::roots::{BatchedBisectionAlgorithms, BatchedBisectionOptions};

/// A reference depth and its square root, taken once for repeated curve evaluations.
pub(super) struct Rooted<'a, R: Runtime<DType = DType>> {
    reference: &'a Tensor<R>,
    /// `sqrt(reference)`.
    depth: Tensor<R>,
}

impl<'a, R: Runtime<DType = DType>> Rooted<'a, R> {
    pub(super) fn new<C: UnaryOps<R>>(
        client: &C,
        reference: &'a Tensor<R>,
    ) -> numr::error::Result<Self> {
        Ok(Self {
            depth: client.sqrt(reference)?,
            reference,
        })
    }
}

/// [`curve`](super::curve) with `sqrt(h)` and `sqrt(reference)` supplied.
///
/// Every operation matches [`curve`](super::curve) on the same operands, so every bit agrees.
pub(super) fn rooted_curve<R, C>(
    client: &C,
    h: &Tensor<R>,
    root_h: &Tensor<R>,
    reference: &Rooted<'_, R>,
    gravity: f64,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let Rooted {
        reference,
        depth: root_reference,
    } = reference;
    let rarefaction =
        client.mul_scalar(&client.sub(root_h, root_reference)?, 2.0 * gravity.sqrt())?;
    let shock_h = client.maximum(h, reference)?;
    let factor = client.sqrt(&client.mul_scalar(
        &client.add_scalar(&client.div(reference, &shock_h)?, 1.0)?,
        0.5,
    )?)?;
    let shock = client.mul(
        &client.mul_scalar(
            &client.div(&client.sub(&shock_h, reference)?, root_reference)?,
            gravity.sqrt(),
        )?,
        &factor,
    )?;
    client.where_cond(&client.le(h, reference)?, &rarefaction, &shock)
}

/// `fL(h) + fR(h) + jump`, taking `sqrt(h)` once for both curves.
fn residual<R, C>(
    client: &C,
    h: &Tensor<R>,
    left: &Rooted<'_, R>,
    right: &Rooted<'_, R>,
    jump: &Tensor<R>,
    gravity: f64,
) -> numr::error::Result<Tensor<R>>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ScalarOps<R>
        + UnaryOps<R>,
{
    let root_h = client.sqrt(h)?;
    let result = client.add(
        &client.add(
            &rooted_curve(client, h, &root_h, left, gravity)?,
            &rooted_curve(client, h, &root_h, right, gravity)?,
        )?,
        jump,
    )?;
    Ok(result)
}

pub(super) fn root_upper<R, C>(
    client: &C,
    hl: &Tensor<R>,
    hr: &Tensor<R>,
    jump: &Tensor<R>,
    solve: &Tensor<R>,
    gravity: f64,
    config: &SolverConfig,
) -> Result<Tensor<R>, EngineError>
where
    R: Runtime<DType = DType>,
    C: RuntimeClient<R>
        + BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + UnaryOps<R>
        + BatchedBisectionAlgorithms<R>,
{
    let zero = constant(hl, 0.0)?;
    // The reference roots stay fixed over every bracket expansion and bisection callback.
    let left = Rooted::new(client, hl)?;
    let right = Rooted::new(client, hr)?;
    let maximum = client.maximum(hl, hr)?;
    let coefficient_scale = gravity.sqrt() * std::f64::consts::FRAC_1_SQRT_2;
    let coefficient = client.add(
        &client.div(&constant(hl, coefficient_scale)?, &left.depth)?,
        &client.div(&constant(hr, coefficient_scale)?, &right.depth)?,
    )?;
    check_finite(client, &coefficient, "compression bracket coefficient")?;
    // For H>=M, fL(H)+fR(H)>=(H-M)*sum(sqrt(g/(2ha))).
    let compression = client.maximum(&client.neg(jump)?, &zero)?;
    let seed = client.add(&maximum, &client.div(&compression, &coefficient)?)?;
    // Each one-sided shock lower bound supplies an upper root bracket, including mixed-wave states.
    let mut seed = seed;
    for (other, reference) in [(&left, &right), (&right, &left)] {
        let numerator = client.maximum(
            &client.sub(
                &client.mul_scalar(&other.depth, 2.0 * gravity.sqrt())?,
                jump,
            )?,
            &zero,
        )?;
        let factor =
            client.mul_scalar(&reference.depth, std::f64::consts::SQRT_2 / gravity.sqrt())?;
        let candidate = client.add(reference.reference, &client.mul(&numerator, &factor)?)?;
        let finite = client.le(&client.abs(&candidate)?, &constant(&candidate, f64::MAX)?)?;
        let candidate = client.where_cond(&finite, &candidate, &seed)?;
        seed = client.minimum(&seed, &candidate)?;
    }
    let hi = client.add_scalar(
        &client.mul_scalar(&seed, 1.0 + 16.0 * f64::EPSILON)?,
        f64::from_bits(1),
    )?;
    check_finite(client, &hi, "compression bracket seed")?;
    // Film-depth wave speeds require relative root accuracy beyond an absolute depth ceiling.
    let options = BatchedBisectionOptions {
        atol: f64::from_bits(1),
        rtol: config.root_relative_depth.min(1e-12),
        max_iter: config.root_iterations,
        max_expansions: config.root_iterations,
        expansion_factor: 2.0,
    };
    let result = client
        .bisect_batched(
            |depth| {
                client.where_cond(
                    solve,
                    &residual(client, depth, &left, &right, jump, gravity)?,
                    &zero,
                )
            },
            &zero,
            &hi,
            &options,
        )
        .map_err(|error| {
            let reason = match &error {
                OptimizeError::DidNotConverge { .. } => format!("iteration limit reached: {error}"),
                _ => error.to_string(),
            };
            EngineError::rejected("Riemann root", reason)
        })?;
    Ok(result.upper)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::tensor_wave::fixture::{
        DEPTHS, GRAVITY, TestResult, physical_upper, setup,
    };
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    #[test]
    fn unequal_positive_films_keep_a_tight_nonnegative_root_bracket() -> TestResult {
        let (client, left, right) = setup()?;
        let zero = constant(&left.h, 0.0)?;
        let solve = constant(&left.h, 1.0)?;
        let upper = root_upper(
            &client,
            &left.h,
            &right.h,
            &zero,
            &solve,
            GRAVITY,
            &SolverConfig::default(),
        )?;
        let (hl, hr) = (
            Rooted::new(&client, &left.h)?,
            Rooted::new(&client, &right.h)?,
        );
        let residuals = residual(&client, &upper, &hl, &hr, &zero, GRAVITY)?.try_to_vec::<f64>()?;
        for ((&(hl, hr), upper), residual) in
            DEPTHS.iter().zip(upper.try_to_vec::<f64>()?).zip(residuals)
        {
            let deep = hl.max(hr);
            let shallow = hl.min(hr);
            let scale = deep.sqrt() * shallow.sqrt();
            let normalized_upper = upper / scale;
            assert!(upper.is_finite() && upper > shallow && upper < deep);
            assert!(
                (2.0..3.0).contains(&normalized_upper),
                "H/sqrt(hl*hr)={normalized_upper}"
            );
            assert!(
                residual.is_finite() && residual >= 0.0,
                "residual={residual}"
            );
            assert!((upper / physical_upper(deep, shallow) - 1.0).abs() <= 2e-12);
        }
        Ok(())
    }

    #[test]
    fn compressed_states_skip_a_nonfinite_one_sided_candidate() -> TestResult {
        let (hl_value, hr_value, gravity, jump_value) =
            (1e300_f64, 1e-200_f64, 1e-300_f64, -1e100_f64);
        let numerator_l = (2.0 * gravity.sqrt() * hr_value.sqrt() - jump_value).max(0.0);
        let factor_l = hl_value.sqrt() * (std::f64::consts::SQRT_2 / gravity.sqrt());
        let candidate_l = hl_value + numerator_l * factor_l;
        assert_eq!(candidate_l, f64::INFINITY);

        let device = CpuDevice::new();
        let client =
            CpuClient::new(device.clone()).with_parallelism(ParallelismConfig::new(Some(1), None));
        let hl = Tensor::<CpuRuntime>::from_slice(&[hl_value], &[1], &device)?;
        let hr = Tensor::<CpuRuntime>::from_slice(&[hr_value], &[1], &device)?;
        let jump = Tensor::<CpuRuntime>::from_slice(&[jump_value], &[1], &device)?;
        let solve = constant(&hl, 1.0)?;
        let upper = root_upper(
            &client,
            &hl,
            &hr,
            &jump,
            &solve,
            gravity,
            &SolverConfig::default(),
        )?;
        let value = upper.item::<f64>()?;
        let (left, right) = (Rooted::new(&client, &hl)?, Rooted::new(&client, &hr)?);
        let residual = residual(&client, &upper, &left, &right, &jump, gravity)?.item::<f64>()?;
        let expected = std::f64::consts::SQRT_2 * 1e150;
        assert!(value.is_finite() && value > 0.0);
        assert!(
            residual.is_finite() && residual >= 0.0,
            "residual={residual}"
        );
        assert!((value / expected - 1.0).abs() <= 2e-12, "upper={value}");
        Ok(())
    }
}
