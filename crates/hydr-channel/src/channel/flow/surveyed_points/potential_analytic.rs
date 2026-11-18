use super::analytic_bottom::AnalyticSource;
use crate::channel::flow::surveyed::{
    PrismaticRiemannRejection,
    equation_ops::reject_mask,
    error::{Arithmetic, HydraulicResult},
};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps,
    TwoSumOps,
};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct AnalyticDifference<R: Runtime<DType = DType>> {
    pub(super) eligible: Tensor<R>,
    pub(super) retained_increment: CertifiedExpansion<R>,
    pub(super) physical_increment: Interval<R>,
}

/// The caller proves common model, owner, bottom band, and gravity for both certificates and widths.
/// Eligibility masks alone do not establish that authority. Delta area retains the complete target-minus-source expression.
pub(super) fn difference<R>(
    a: &mut Arithmetic<'_, R>,
    source: &AnalyticSource<R>,
    target: &AnalyticSource<R>,
    source_width: &Interval<R>,
    target_width: &Interval<R>,
    delta_area: &CertifiedExpansion<R>,
    gravity: f64,
) -> HydraulicResult<AnalyticDifference<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = a.client();
    let zero = Tensor::zeros(delta_area.high().shape(), DType::F64, c.device())?;
    let one = Tensor::ones(delta_area.high().shape(), DType::F64, c.device())?;
    let rectangular = c.mul(&source.rectangular, &target.rectangular)?;
    let pointed = c.mul(&source.pointed, &target.pointed)?;
    let eligible = c.mul(
        &c.mul(&source.eligible, &target.eligible)?,
        &c.maximum(&rectangular, &pointed)?,
    )?;
    let exact_zero = c.mul(
        &c.mul(
            &c.eq(delta_area.high(), &zero)?,
            &c.eq(delta_area.low(), &zero)?,
        )?,
        &c.eq(delta_area.abs_error(), &zero)?,
    )?;
    let active = c.mul(&eligible, &c.eq(&exact_zero, &zero)?)?;
    let rectangular = c.mul(&rectangular, &active)?;
    let pointed = c.mul(&pointed, &active)?;
    let source_phi = masked(a, &active, &source.potential, &zero, &zero)?;
    let target_phi = masked(a, &active, &target.potential, &zero, &zero)?;
    let sum = a.expansion_add(&source_phi, &target_phi)?;
    let sum = masked(a, &active, &sum, &one, &zero)?;
    let sum_bounds = a.expansion_interval(&sum)?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(sum_bounds.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic potential difference lacks a positive potential sum. Tighten the endpoint certificates",
    )?;

    let source_t = interval_masked(a, &pointed, source_width, &zero)?;
    let target_t = interval_masked(a, &pointed, target_width, &zero)?;
    let pointed_width = a.add(&source_t, &target_t)?;
    let rectangular_source = interval_masked(a, &rectangular, source_width, &one)?;
    let rectangular_target = interval_masked(a, &rectangular, target_width, &one)?;
    let rectangular_width = a.bounds(
        &c.maximum(rectangular_source.lower(), rectangular_target.lower())?,
        &c.minimum(rectangular_source.upper(), rectangular_target.upper())?,
    )?;
    let width = a.bounds(
        &c.where_cond(&pointed, pointed_width.lower(), rectangular_width.lower())?,
        &c.where_cond(&pointed, pointed_width.upper(), rectangular_width.upper())?,
    )?;
    reject_mask(
        a,
        &c.mul(&active, &c.le(width.lower(), &zero)?)?,
        PrismaticRiemannRejection::QuadratureUnresolved,
        "Analytic potential difference lacks a positive width denominator. Tighten the endpoint certificates",
    )?;
    let width = a.expansion_from_interval(&width)?;
    let delta = masked(a, &active, delta_area, &zero, &zero)?;
    let numerator = a.expansion_div_positive(&delta, &width)?;
    let gravity = a.expansion_point(&c.mul_scalar(&one, gravity)?)?;
    let numerator = a.expansion_mul(&numerator, &gravity)?;
    let four = c.mul_scalar(&one, 4.0)?;
    let sixteen = c.mul_scalar(&one, 16.0)?;
    let factor = a.expansion_point(&c.where_cond(&pointed, &sixteen, &four)?)?;
    let numerator = a.expansion_mul(&numerator, &factor)?;
    let retained_increment = a.expansion_div_positive(&numerator, &sum)?;
    let projected = a.expansion_interval(&retained_increment)?;
    let numerator = a.expansion_interval(&numerator)?;
    let direct = a.div_positive_interval(&numerator, &sum_bounds)?;
    let delta = a.expansion_interval(&delta)?;
    let lower = c.maximum(projected.lower(), direct.lower())?;
    let upper = c.minimum(projected.upper(), direct.upper())?;
    let physical_increment = a.bounds(
        &c.where_cond(
            &c.ge(delta.lower(), &zero)?,
            &c.maximum(&lower, &zero)?,
            &lower,
        )?,
        &c.where_cond(
            &c.le(delta.upper(), &zero)?,
            &c.minimum(&upper, &zero)?,
            &upper,
        )?,
    )?;
    Ok(AnalyticDifference {
        eligible,
        retained_increment,
        physical_increment,
    })
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

fn interval_masked<R>(
    a: &mut Arithmetic<'_, R>,
    active: &Tensor<R>,
    value: &Interval<R>,
    fallback: &Tensor<R>,
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
    Ok(a.bounds(
        &c.where_cond(active, value.lower(), fallback)?,
        &c.where_cond(active, value.upper(), fallback)?,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::error;
    use crate::channel::flow::surveyed_points::analytic_bottom;
    use crate::channel::flow::surveyed_points::source_fixture_support::{
        self as support, Client, Result, batch, components, endpoints, expansion, hydraulic, zero,
    };
    use crate::channel::section::{HydraulicGeometry, PrismaticSectionBatch};

    struct Case {
        source: [f64; 2],
        target: [f64; 2],
        source_depth: [f64; 2],
        target_depth: [f64; 2],
        source_width: [f64; 2],
        target_width: [f64; 2],
    }

    struct Observed {
        eligible: Vec<f64>,
        retained: Vec<[u64; 3]>,
        physical: Vec<[f64; 2]>,
        source_phi: Vec<[u64; 3]>,
        target_phi: Vec<[u64; 3]>,
    }

    fn evaluate<R>(
        client: &R::Client,
        section: &PrismaticSectionBatch<R>,
        owner: usize,
        cases: &[Case],
        gravity: f64,
        uncertain_delta: bool,
        strided: bool,
    ) -> Result<Observed>
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let n = cases.len();
        let sources = expansion::<R>(
            client,
            &cases
                .iter()
                .map(|case| [case.source[0].to_bits(), case.source[1].to_bits(), 0])
                .collect::<Vec<_>>(),
            strided,
        )?;
        let targets = expansion::<R>(
            client,
            &cases
                .iter()
                .map(|case| [case.target[0].to_bits(), case.target[1].to_bits(), 0])
                .collect::<Vec<_>>(),
            strided,
        )?;
        let owners = Tensor::from_slice(&vec![i64::try_from(owner)?; n], &[n], client.device())?;
        let geometry = HydraulicGeometry::from_batch(section);
        let (difference, source, target) = hydraulic(error::checked::<R, _>(
            client,
            "analytic signed potential differences",
            |a| {
                let bands = geometry.gather(a, &owners)?;
                let interval = |a: &mut error::Arithmetic<'_, R>, field: fn(&Case) -> [f64; 2]| {
                    let lower = Tensor::from_slice(
                        &cases.iter().map(|case| field(case)[0]).collect::<Vec<_>>(),
                        &[n],
                        client.device(),
                    )?;
                    let upper = Tensor::from_slice(
                        &cases.iter().map(|case| field(case)[1]).collect::<Vec<_>>(),
                        &[n],
                        client.device(),
                    )?;
                    a.bounds(&lower, &upper)
                };
                let source_depth = interval(a, |case| case.source_depth)?;
                let target_depth = interval(a, |case| case.target_depth)?;
                let source = analytic_bottom::source(a, &bands, &sources, &source_depth, gravity)?;
                let target = analytic_bottom::source(a, &bands, &targets, &target_depth, gravity)?;
                let source_width = interval(a, |case| case.source_width)?;
                let target_width = interval(a, |case| case.target_width)?;
                let delta = a.expansion_sub(&targets, &sources)?;
                let delta = if uncertain_delta {
                    let radius =
                        Tensor::from_slice(&vec![2.0_f64.powi(-80); n], &[n], client.device())?;
                    a.expansion_bounds(delta.high(), delta.low(), &radius)?
                } else {
                    delta
                };
                let difference = difference(
                    a,
                    &source,
                    &target,
                    &source_width,
                    &target_width,
                    &delta,
                    gravity,
                )?;
                Ok((difference, source, target))
            },
        ))?;
        Ok(Observed {
            eligible: difference.eligible.try_to_vec::<f64>()?,
            retained: components(&difference.retained_increment)?,
            physical: endpoints(&difference.physical_increment)?,
            source_phi: components(&source.potential)?,
            target_phi: components(&target.potential)?,
        })
    }

    fn exact_case(
        source: f64,
        target: f64,
        source_depth: f64,
        target_depth: f64,
        source_width: f64,
        target_width: f64,
    ) -> Case {
        Case {
            source: [source, 0.0],
            target: [target, 0.0],
            source_depth: [source_depth; 2],
            target_depth: [target_depth; 2],
            source_width: [source_width; 2],
            target_width: [target_width; 2],
        }
    }

    fn matrix<R>(client: R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let section = batch::<R>(&client)?;
        let owner = |id: &str| {
            support::cases()
                .iter()
                .find(|case| case.id == id)
                .map(|case| case.owner)
                .ok_or("Missing frozen source owner")
        };
        let rectangle = owner("scalar_rectangle_uniform_rest")?;
        let triangle = owner("scalar_triangle_uniform_flow")?;
        let trapezoid = owner("scalar_trapezoid_uniform_flow")?;
        let rectangle_cases = [
            exact_case(1.0, 4.0, 1.0, 4.0, 1.0, 1.0),
            exact_case(4.0, 1.0, 4.0, 1.0, 1.0, 1.0),
            exact_case(0.0, 0.0, 0.0, 0.0, 1.0, 1.0),
        ];
        let pointed_cases = [
            exact_case(1.0, 16.0, 1.0, 4.0, 2.0, 8.0),
            exact_case(16.0, 1.0, 4.0, 1.0, 8.0, 2.0),
            exact_case(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        for strided in [false, true] {
            for (owner, cases, gravity, expected) in [
                (rectangle, rectangle_cases.as_slice(), 1.0, [2.0, -2.0, 0.0]),
                (triangle, pointed_cases.as_slice(), 2.0, [4.0, -4.0, 0.0]),
            ] {
                let observed =
                    evaluate::<R>(&client, &section, owner, cases, gravity, false, strided)?;
                assert_eq!(observed.eligible, [1.0; 3]);
                for (index, expected) in expected.into_iter().enumerate() {
                    assert_eq!(
                        observed.retained[index].map(f64::from_bits),
                        [expected, 0.0, 0.0]
                    );
                    assert_eq!(observed.physical[index], [expected; 2]);
                }
            }
            let unit = Tensor::from_slice(&[1.0], &[1], client.device())?;
            let next = client.next_up(&unit)?.item::<f64>()?;
            let width = Tensor::from_slice(&[2.0], &[1], client.device())?;
            let next_width = client.next_up(&width)?.item::<f64>()?;
            let epsilon = 2.0_f64.powi(-80);
            let low = client
                .next_down(&Tensor::from_slice(&[epsilon], &[1], client.device())?)?
                .item::<f64>()?;
            for (owner, gravity, width, widened_width) in [
                (rectangle, 1.0, 1.0, [1.0, 1.0]),
                (triangle, 2.0, 2.0, [2.0, next_width]),
            ] {
                let forward = Case {
                    source: [1.0, 0.0],
                    target: [1.0, epsilon],
                    source_depth: [1.0; 2],
                    target_depth: [1.0, next],
                    source_width: [width; 2],
                    target_width: widened_width,
                };
                let reverse = Case {
                    source: forward.target,
                    target: forward.source,
                    source_depth: forward.target_depth,
                    target_depth: forward.source_depth,
                    source_width: forward.target_width,
                    target_width: forward.source_width,
                };
                let observed = evaluate::<R>(
                    &client,
                    &section,
                    owner,
                    &[forward, reverse],
                    gravity,
                    false,
                    strided,
                )?;
                assert_eq!(observed.eligible, [1.0; 2]);
                // Concavity gives epsilon/(1+epsilon) <= DeltaPhi <= epsilon for both controls.
                for (index, expected) in [[low, epsilon], [-epsilon, -low]].into_iter().enumerate()
                {
                    let actual = observed.physical[index];
                    assert!(
                        actual[0] <= expected[0] && expected[1] <= actual[1],
                        "{actual:?} excludes {expected:?}"
                    );
                    let error = (actual[0] - expected[1])
                        .abs()
                        .max((actual[1] - expected[0]).abs());
                    assert!(
                        error <= epsilon * 1e-10,
                        "weak signed increment error={error}"
                    );
                    assert!(f64::from_bits(observed.retained[index][2]) > 0.0);
                }
            }
            let same = [exact_case(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)];
            let uncertain = evaluate::<R>(&client, &section, rectangle, &same, 1.0, true, strided)?;
            assert!(f64::from_bits(uncertain.retained[0][2]) >= epsilon);
            assert!(uncertain.physical[0][0] < 0.0 && uncertain.physical[0][1] > 0.0);
            let unavailable = [Case {
                source: [1.0, 0.0],
                target: [2.0, 0.0],
                source_depth: [0.0, 2.0],
                target_depth: [0.0, 2.0],
                source_width: [0.0; 2],
                target_width: [0.0; 2],
            }];
            let inactive = evaluate::<R>(
                &client,
                &section,
                trapezoid,
                &unavailable,
                1.0,
                false,
                strided,
            )?;
            assert_eq!(inactive.eligible, [0.0]);
            zero(inactive.retained[0], "unavailable general bottom");
            assert_eq!(inactive.physical[0], [0.0, 0.0]);
        }
        film::<R>(&client, &section, rectangle)
    }

    fn film<R>(client: &R::Client, section: &PrismaticSectionBatch<R>, owner: usize) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R>,
    {
        let original = support::cases()
            .iter()
            .find(|case| case.id == "retained_vacuum_inside_film")
            .ok_or("Missing original film source")?;
        let epsilon = -f64::from_bits(original.inputs[3][1]);
        let area = epsilon * epsilon / 16.0;
        let case = exact_case(1.0, area, 1.0, area, 1.0, 1.0);
        let observed = evaluate::<R>(client, section, owner, &[case], 1.0, false, false)?;
        assert_eq!(observed.source_phi[0].map(f64::from_bits), [2.0, 0.0, 0.0]);
        assert_eq!(
            observed.target_phi[0].map(f64::from_bits),
            [epsilon / 2.0, 0.0, 0.0]
        );
        assert_eq!(
            observed.retained[0].map(f64::from_bits),
            [-2.0, epsilon / 2.0, 0.0]
        );
        let dry = evaluate::<R>(
            client,
            section,
            owner,
            &[exact_case(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)],
            1.0,
            false,
            false,
        )?;
        zero(dry.retained[0], "structural dry equality");
        Ok(())
    }

    #[test]
    fn configured_cpu_analytic_potential_differences_retain_signed_changes_and_uncertainty()
    -> Result {
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
    fn cuda_analytic_potential_differences_retain_signed_changes_and_uncertainty() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        matrix::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
    }
}
