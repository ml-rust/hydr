use super::analytic_bottom::AnalyticSource;
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    equation_ops::reject_mask,
    error::{Arithmetic, HydraulicResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct AnalyticSonic<R: Runtime<DType = DType>> {
    pub(super) eligible: Tensor<R>,
    pub(super) retained_residual: CertifiedExpansion<R>,
    pub(super) physical_residual: Interval<R>,
}

/// The caller proves common model, owner, bottom band, and gravity for the target and outgoing invariant.
/// Epsilon is positive except for structural dry equality. Matching masks or overlapping certificates prove no authority.
pub(super) fn residual<R>(
    a: &mut Arithmetic<'_, R>,
    target: &AnalyticSource<R>,
    epsilon: &CertifiedExpansion<R>,
) -> HydraulicResult<AnalyticSonic<R>>
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
    let zero = Tensor::zeros(epsilon.high().shape(), a.dtype(), c.device())?;
    let one = Tensor::ones(epsilon.high().shape(), a.dtype(), c.device())?;
    let eligible = c.mul(
        &target.eligible,
        &c.maximum(&target.rectangular, &target.pointed)?,
    )?;
    let epsilon_zero = exact_zero::<R>(a, epsilon, &zero)?;
    let sound_zero = exact_zero::<R>(a, &target.sound_squared, &zero)?;
    let structural_zero = c.mul(&epsilon_zero, &sound_zero)?;
    let active = c.mul(&eligible, &c.eq(&structural_zero, &zero)?)?;
    let epsilon = masked(a, &active, epsilon, &zero, &zero)?;
    let epsilon_bounds = a.expansion_interval(&epsilon)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(epsilon_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic sonic invariant lacks a positive lower bound. Tighten its complete source certificate",
    )?;
    let sound = masked(a, &active, &target.sound, &zero, &zero)?;
    let squared = masked(a, &active, &target.sound_squared, &zero, &zero)?;
    let three = c.mul_scalar(&one, 3.0)?;
    let five = c.mul_scalar(&one, 5.0)?;
    let nine = c.mul_scalar(&one, 9.0)?;
    let twenty_five = c.mul_scalar(&one, 25.0)?;
    let k = a.expansion_point(&c.where_cond(&target.pointed, &five, &three)?)?;
    let k_squared = a.expansion_point(&c.where_cond(&target.pointed, &twenty_five, &nine)?)?;
    let scaled_squared = a.expansion_mul(&k_squared, &squared)?;
    let epsilon_squared = a.expansion_mul(&epsilon, &epsilon)?;
    let numerator = a.expansion_sub(&scaled_squared, &epsilon_squared)?;
    let scaled_sound = a.expansion_mul(&k, &sound)?;
    let denominator = a.expansion_add(&scaled_sound, &epsilon)?;
    let denominator = masked(a, &active, &denominator, &one, &zero)?;
    let denominator_bounds = a.expansion_interval(&denominator)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(denominator_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic sonic denominator lacks a positive lower bound. Tighten the complete target and invariant certificates",
    )?;
    let retained_residual = a.expansion_div_positive(&numerator, &denominator)?;
    let projected = a.expansion_interval(&retained_residual)?;
    let numerator = a.expansion_interval(&numerator)?;
    let direct = a.div_positive_interval(&numerator, &denominator_bounds)?;
    let lower = c.maximum(projected.lower(), direct.lower())?;
    let upper = c.minimum(projected.upper(), direct.upper())?;
    let physical_residual = a.bounds(
        &c.where_cond(
            &c.ge(numerator.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?,
        &c.where_cond(
            &c.le(numerator.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?,
    )?;
    Ok(AnalyticSonic {
        eligible,
        retained_residual,
        physical_residual,
    })
}

fn exact_zero<R>(
    a: &Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
    zero: &Tensor<R>,
) -> HydraulicResult<Tensor<R>>
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
    Ok(c.mul(
        &c.mul(&c.eq(value.high(), zero)?, &c.eq(value.low(), zero)?)?,
        &c.eq(value.abs_error(), zero)?,
    )?)
}

fn masked<R>(
    a: &mut Arithmetic<'_, R>,
    active: &Tensor<R>,
    value: &CertifiedExpansion<R>,
    fallback: &Tensor<R>,
    zero: &Tensor<R>,
) -> HydraulicResult<CertifiedExpansion<R>>
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
    Ok(a.expansion_bounds(
        &c.where_cond(active, value.high(), fallback)?,
        &c.where_cond(active, value.low(), zero)?,
        &c.where_cond(active, value.abs_error(), zero)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::error;
    use crate::channel::flow::surveyed_points::source_fixture_support::{
        self as support, Client, Result, batch, components, endpoints, expansion, hydraulic, zero,
    };
    use crate::channel::flow::surveyed_points::{analytic_bottom, source_velocity};
    use crate::channel::section::{HydraulicGeometry, PrismaticSectionBatch};

    fn film<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let original = support::cases()
            .iter()
            .find(|case| case.id == "retained_vacuum_inside_film")
            .ok_or("Missing original film source")?;
        let geometry = HydraulicGeometry::from_batch(section);
        let epsilon_value = -f64::from_bits(original.inputs[3][1]);
        // Frozen original-film directed parameter endpoints and independent residual enclosures.
        let parameters = [0x3fbc71c71c71c71c_u64, 0x3fbc71c71c71c71d];
        let references = [
            [0xb780000000000001_u64, 0xb780000000000000],
            [0x3783ffffffffffff_u64, 0x3784000000000000],
        ];
        let allowance = f64::from_bits(0x38db7cdfd9d7bdba);
        for strided in [false, true] {
            let owners =
                Tensor::from_slice(&[i64::try_from(original.owner)?; 2], &[2], client.device())?;
            let inputs = (0..4)
                .map(|field| expansion::<R>(client, &[original.inputs[field]; 2], strided))
                .collect::<Result<Vec<_>>>()?;
            let captured =
                hydraulic(section.capture_riemann_points(
                    &owners, &inputs[0], &inputs[1], &inputs[2], &inputs[3],
                ))?;
            let velocity = hydraulic(source_velocity::prepare::<R>(client, &captured))?;
            let (result, epsilon, squared_error, left, right, target) = hydraulic(
                error::checked::<R, _>(client, "retained original-film right sonic bracket", |a| {
                    let bands = geometry.gather(a, &captured.owners)?;
                    let one = Tensor::ones(&[2], DType::F64, client.device())?;
                    let source_depth = a.point(&one)?;
                    let left = analytic_bottom::source(
                        a,
                        &bands,
                        &captured.left_area,
                        &source_depth,
                        1.0,
                    )?;
                    let right = analytic_bottom::source(
                        a,
                        &bands,
                        &captured.right_area,
                        &source_depth,
                        1.0,
                    )?;
                    let source_sum = a.expansion_add(&left.potential, &right.potential)?;
                    let epsilon = a.expansion_sub(&source_sum, &velocity.difference)?;
                    let scale = a.expansion_mul(&epsilon, &epsilon)?;
                    let parameter = a.expansion_point(&Tensor::from_slice(
                        &parameters.map(f64::from_bits),
                        &[2],
                        client.device(),
                    )?)?;
                    let area = a.expansion_mul(&scale, &parameter)?;
                    let depth = a.expansion_interval(&area)?;
                    let target = analytic_bottom::source(a, &bands, &area, &depth, 1.0)?;
                    let squared_error = a.expansion_sub(&target.sound_squared, &area)?;
                    let result = residual(a, &target, &epsilon)?;
                    Ok((result, epsilon, squared_error, left, right, target))
                }),
            )?;
            assert_eq!(result.eligible.try_to_vec::<f64>()?, [1.0; 2]);
            assert_eq!(components(&epsilon)?, [[epsilon_value.to_bits(), 0, 0]; 2]);
            for value in [&left, &right] {
                assert_eq!(
                    components(&value.potential)?,
                    [[2.0_f64.to_bits(), 0, 0]; 2]
                );
                assert_eq!(components(&value.sound)?, [[1.0_f64.to_bits(), 0, 0]; 2]);
            }
            for point in components(&squared_error)? {
                zero(point, "retained pre-root rectangular sound square");
            }
            assert!(
                components(&target.sound)?
                    .iter()
                    .all(|point| f64::from_bits(point[2]) > 0.0)
            );
            let physical = endpoints(&result.physical_residual)?;
            assert!(physical[0][1] < 0.0 && physical[1][0] > 0.0, "{physical:?}");
            for (actual, reference) in physical.into_iter().zip(references) {
                let expected = reference.map(f64::from_bits);
                assert!(
                    actual[0] <= expected[0] && expected[1] <= actual[1],
                    "{actual:?} excludes {expected:?}"
                );
                let excess = (actual[0] - expected[1])
                    .abs()
                    .max((actual[1] - expected[0]).abs());
                assert!(excess <= allowance, "epsilon-local residual error={excess}");
            }
            assert!(
                components(&result.retained_residual)?
                    .iter()
                    .all(|point| f64::from_bits(point[2]) >= 0.0)
            );
        }
        Ok(())
    }

    fn controls<R>(client: &R::Client, section: &PrismaticSectionBatch<R>) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let geometry = HydraulicGeometry::from_batch(section);
        for (id, gravity, k) in [
            ("scalar_rectangle_uniform_rest", 1.0, 3.0),
            ("scalar_triangle_uniform_flow", 2.0, 5.0),
        ] {
            let owner = support::cases()
                .iter()
                .find(|case| case.id == id)
                .map(|case| case.owner)
                .ok_or("Missing analytic sonic control owner")?;
            let owner = i64::try_from(owner)?;
            let owners = Tensor::from_slice(&[owner; 4], &[4], client.device())?;
            let (result, uncertain, widened) = hydraulic(error::checked::<R, _>(
                client,
                "dyadic analytic sonic controls",
                |a| {
                    let bands = geometry.gather(a, &owners)?;
                    let values = Tensor::from_slice(&[1.0, 1.0, 1.0, 0.0], &[4], client.device())?;
                    let area = a.expansion_point(&values)?;
                    let depth = a.point(&values)?;
                    let target = analytic_bottom::source(a, &bands, &area, &depth, gravity)?;
                    let epsilon = a.expansion_point(&Tensor::from_slice(
                        &[k - 1.0, k, k + 1.0, 0.0],
                        &[4],
                        client.device(),
                    )?)?;
                    let result = residual(a, &target, &epsilon)?;

                    let one_owner = Tensor::from_slice(&[owner], &[1], client.device())?;
                    let bands = geometry.gather(a, &one_owner)?;
                    let one = Tensor::ones(&[1], DType::F64, client.device())?;
                    let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                    let area = a.expansion_point(&one)?;
                    let depth = a.point(&one)?;
                    let mut target = analytic_bottom::source(a, &bands, &area, &depth, gravity)?;
                    let high = Tensor::from_slice(&[k], &[1], client.device())?;
                    let radius = Tensor::from_slice(&[2.0_f64.powi(-20)], &[1], client.device())?;
                    let epsilon = a.expansion_bounds(&high, &zero, &radius)?;
                    let uncertain = residual(a, &target, &epsilon)?;
                    let exact_epsilon = a.expansion_point(&high)?;
                    // Widen only a valid squared-sound certificate; the original physical target and root stay unchanged.
                    target.sound_squared = a.expansion_bounds(
                        target.sound_squared.high(),
                        target.sound_squared.low(),
                        &radius,
                    )?;
                    let widened = residual(a, &target, &exact_epsilon)?;
                    Ok((result, uncertain, widened))
                },
            ))?;
            assert_eq!(result.eligible.try_to_vec::<f64>()?, [1.0; 4]);
            for (point, expected) in components(&result.retained_residual)?
                .into_iter()
                .zip([1.0_f64, 0.0, -1.0, 0.0])
            {
                assert_eq!(point.map(f64::from_bits), [expected, 0.0, 0.0]);
            }
            for (actual, expected) in endpoints(&result.physical_residual)?
                .into_iter()
                .zip([1.0_f64, 0.0, -1.0, 0.0])
            {
                assert_eq!(actual, [expected; 2]);
            }
            for observed in [&uncertain, &widened] {
                assert!(f64::from_bits(components(&observed.retained_residual)?[0][2]) > 0.0);
                let interval = endpoints(&observed.physical_residual)?[0];
                assert!(interval[0] < 0.0 && interval[1] > 0.0, "{interval:?}");
            }
        }
        let owner = support::cases()
            .iter()
            .find(|case| case.id == "scalar_trapezoid_uniform_flow")
            .map(|case| case.owner)
            .ok_or("Missing unavailable sonic source owner")?;
        let owners = Tensor::from_slice(&[i64::try_from(owner)?], &[1], client.device())?;
        let unavailable = hydraulic(error::checked::<R, _>(
            client,
            "unavailable analytic sonic operands",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let one = Tensor::ones(&[1], DType::F64, client.device())?;
                let zero = Tensor::zeros(&[1], DType::F64, client.device())?;
                let two = Tensor::from_slice(&[2.0], &[1], client.device())?;
                let area = a.expansion_point(&one)?;
                let depth = a.bounds(&zero, &two)?;
                let target = analytic_bottom::source(a, &bands, &area, &depth, 1.0)?;
                let epsilon =
                    a.expansion_point(&Tensor::from_slice(&[f64::MAX], &[1], client.device())?)?;
                residual(a, &target, &epsilon)
            },
        ))?;
        assert_eq!(unavailable.eligible.try_to_vec::<f64>()?, [0.0]);
        zero(
            components(&unavailable.retained_residual)?[0],
            "unavailable sonic residual",
        );
        assert_eq!(endpoints(&unavailable.physical_residual)?, [[0.0; 2]]);
        Ok(())
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = batch::<R>(&client)?;
        film::<R>(&client, &section)?;
        controls::<R>(&client, &section)
    }

    #[test]
    fn configured_cpu_analytic_sonic_residuals_retain_directed_film_signs() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let original = CpuClient::new(CpuDevice::new());
        let before = original.parallelism();
        let config = ParallelismConfig::new(Some(2), Some(257));
        let client = original.with_parallelism(config);
        let retained = client.clone();
        matrix::<CpuRuntime>(client)?;
        assert_eq!(retained.parallelism(), config);
        assert_eq!(original.parallelism(), before);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_analytic_sonic_residuals_retain_directed_film_signs() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
