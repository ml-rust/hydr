use super::band_lanes::IntegrationLanes;
use super::derivative_data::{CASES, DerivativeCase};
use super::derivatives::{fourth_derivative, panel_fourth_bound};
use super::error::{Arithmetic, HydraulicResult, PrismaticRiemannError, checked, into_public};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;

fn operand<R>(
    a: &mut Arithmetic<'_, R>,
    bits: [u64; 2],
    shape: &[usize],
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
    Ok(a.bounds(
        &Tensor::from_slice(&[f64::from_bits(bits[0])], shape, a.client().device())?,
        &Tensor::from_slice(&[f64::from_bits(bits[1])], shape, a.client().device())?,
    )?)
}

fn lanes<R>(
    a: &mut Arithmetic<'_, R>,
    case: &DerivativeCase,
) -> HydraulicResult<IntegrationLanes<R>>
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
    let one = [1.0_f64.to_bits(); 2];
    let zero = [0.0_f64.to_bits(); 2];
    Ok(IntegrationLanes {
        face_owner: Tensor::from_slice(&[0_i64], &[1], c.device())?,
        band_index: Tensor::from_slice(
            &[if case.bottom { 0_i64 } else { 1_i64 }],
            &[1],
            c.device(),
        )?,
        bottom: Tensor::from_slice(&[if case.bottom { 1.0 } else { 0.0 }], &[1], c.device())?,
        active: Tensor::from_slice(
            &[if case.analytic_zero { 0.0 } else { 1.0 }],
            &[1],
            c.device(),
        )?,
        analytic: Tensor::from_slice(
            &[if case.analytic_zero { 1.0 } else { 0.0 }],
            &[1],
            c.device(),
        )?,
        extent: operand(a, case.extent, &[1])?,
        width: operand(a, if case.analytic_zero { one } else { case.width }, &[1])?,
        slope: operand(a, if case.analytic_zero { zero } else { case.slope }, &[1])?,
        area: operand(a, case.area, &[1])?,
        gravity: operand(a, case.gravity, &[])?,
        face_count: 1,
        band_count: 1,
    })
}

fn independent_references<R>(client: R::Client) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    for case in CASES {
        let lanes = checked::<R, _>(&client, "derivative reference operands", |a| lanes(a, case))
            .inspect_err(|error| {
            eprintln!("Derivative case={}, stage=operands: {error:?}", case.id)
        })?;
        let points: Vec<f64> = case.points.iter().map(|(x, _)| *x).collect();
        let owner = Tensor::from_slice(&[0_i64; 6], &[6], client.device())?;
        let derivative = checked::<R, _>(&client, "pointwise fourth derivative regression", |a| {
            let x = a.point(&Tensor::from_slice(&points, &[6], client.device())?)?;
            fourth_derivative(a, &lanes, &owner, &x)
        })
        .inspect_err(|error| {
            eprintln!("Derivative case={}, stage=pointwise: {error:?}", case.id)
        })?;
        let lower = derivative.lower().try_to_vec::<f64>()?;
        let upper = derivative.upper().try_to_vec::<f64>()?;
        if case.id == "bottom_partial_extent_overlap" {
            // At x=0, f''''=-30*d^(5/2). For 0<=d<=2^-10, the full range is [-30*2^-25,0].
            let exact_lower = -30.0 / 33_554_432.0;
            assert!(
                lower[0] <= exact_lower && upper[0] >= 0.0,
                "{} at x=0: production=[{},{}], exact full range=[{exact_lower},0]",
                case.id,
                lower[0],
                upper[0]
            );
        }
        for (index, (_, reference)) in case.corner_points.iter().enumerate() {
            let reference_lower = f64::from_bits(reference[0]);
            let reference_upper = f64::from_bits(reference[1]);
            assert!(
                lower[index] <= reference_lower && upper[index] >= reference_upper,
                "{} at x={}: production=[{},{}], independent corner hull=[{},{}]",
                case.id,
                points[index],
                lower[index],
                upper[index],
                reference_lower,
                reference_upper
            );
            // Independent whole-box formulas can overestimate differently. Their overlap remains a necessary consistency check.
            let box_lower = f64::from_bits(case.points[index].1[0]);
            let box_upper = f64::from_bits(case.points[index].1[1]);
            assert!(
                lower[index] <= box_upper && upper[index] >= box_lower,
                "{} at x={}: production=[{},{}], independent whole box=[{box_lower},{box_upper}]",
                case.id,
                points[index],
                lower[index],
                upper[index]
            );
            if case.analytic_zero {
                assert_eq!(lower[index], 0.0, "{} lower derivative", case.id);
                assert_eq!(upper[index], 0.0, "{} upper derivative", case.id);
            }
        }
        let left: Vec<f64> = case.panels.iter().map(|(left, _)| *left).collect();
        let right: Vec<f64> = case.panels.iter().map(|(_, right)| *right).collect();
        let owner = Tensor::from_slice(&[0_i64; 3], &[3], client.device())?;
        let bound = checked::<R, _>(&client, "whole-panel fourth derivative recurrence", |a| {
            let x = a.bounds(
                &Tensor::from_slice(&left, &[3], client.device())?,
                &Tensor::from_slice(&right, &[3], client.device())?,
            )?;
            panel_fourth_bound(a, &lanes, &owner, &x)
        })
        .inspect_err(|error| {
            eprintln!("Derivative case={}, stage=whole-panel: {error:?}", case.id)
        })?
        .try_to_vec::<f64>()?;
        for (panel, &(left, right)) in case.panels.iter().enumerate() {
            assert!(
                bound[panel].is_finite() && bound[panel] >= 0.0,
                "{} panel={panel}",
                case.id
            );
            if case.analytic_zero {
                assert_eq!(bound[panel], 0.0, "{} analytic panel", case.id);
            }
            // Corners provide regression evidence. The full-parameter Taylor recurrence proves the complete panel bound.
            for &(x, reference) in &case.corner_points {
                if left <= x && x <= right {
                    let absolute = f64::from_bits(reference[0])
                        .abs()
                        .max(f64::from_bits(reference[1]).abs());
                    assert!(
                        bound[panel] >= absolute,
                        "{} panel={panel}, x={x}: M4={} < independent={absolute}",
                        case.id,
                        bound[panel]
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn fourth_derivatives_enclose_independent_logarithmic_references_cpu()
-> Result<(), PrismaticRiemannError> {
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
    let client =
        CpuClient::new(CpuDevice::new()).with_parallelism(ParallelismConfig::new(Some(1), None));
    independent_references::<CpuRuntime>(client)
        .map_err(|error| into_public(error, Default::default()))
}

#[cfg(feature = "cuda")]
#[test]
fn fourth_derivatives_enclose_independent_logarithmic_references_cuda()
-> Result<(), Box<dyn std::error::Error>> {
    use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
    let client = CudaClient::new(CudaDevice::new(0))?;
    independent_references::<CudaRuntime>(client)
        .map_err(|error| into_public(error, Default::default()))?;
    Ok(())
}

#[test]
fn simpson_runtime_errors_retain_their_diagnostics() {
    use super::error::HydraulicError;
    use super::integration::integration_error;
    use solvr::integrate::IntegrateError;
    let diagnostic = "configured CUDA client rejected allocation";
    let mapped = integration_error(IntegrateError::NumrError(diagnostic.into()));
    assert!(
        matches!(mapped, HydraulicError::Runtime(numr::error::Error::Backend(context)) if context == diagnostic)
    );
}

#[test]
fn simpson_query_budget_errors_retain_the_work_limit() {
    use super::PrismaticRiemannRejection;
    use super::error::HydraulicError;
    use super::integration::integration_error;
    use solvr::integrate::IntegrateError;
    let diagnostic = "Certified Simpson needs 26 queries but max_points=25";
    let mapped = integration_error(IntegrateError::InvalidParameter {
        parameter: "max_points".into(),
        message: diagnostic.into(),
    });
    assert!(
        matches!(mapped, HydraulicError::Rejected { reason: PrismaticRiemannRejection::WorkLimit, context, .. } if context.contains(diagnostic))
    );
}

#[test]
fn simpson_fatal_arithmetic_errors_precede_callback_causes() {
    use super::PrismaticRiemannRejection;
    use super::error::HydraulicError;
    use super::integration::complete_integration;
    use solvr::common::arithmetic::ArithmeticError;
    use solvr::integrate::IntegrateError;
    for actual in [
        IntegrateError::ArithmeticRange {
            stage: "combined Simpson arithmetic",
        },
        IntegrateError::UnsupportedArithmetic {
            dtype: DType::F64,
            failed_probes: vec!["provider_contract"],
        },
        IntegrateError::NumrError("configured client rejected reduction".into()),
    ] {
        let callback = HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::QuadratureUnresolved,
            query: None,
            section_owner: None,
            context: "retained callback cause".into(),
        };
        let mapped = complete_integration::<()>(Err(actual), Some(callback));
        assert!(matches!(
            mapped,
            Err(HydraulicError::Arithmetic(
                ArithmeticError::Range { .. } | ArithmeticError::UnsupportedArithmetic { .. }
            )) | Err(HydraulicError::Runtime(_))
        ));
    }
}

#[test]
fn simpson_callback_sentinel_restores_its_typed_cause() {
    use super::PrismaticRiemannRejection;
    use super::error::HydraulicError;
    use super::integration::complete_integration;
    use solvr::integrate::IntegrateError;
    let callback = HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::WorkLimit,
        query: None,
        section_owner: None,
        context: "retained callback work limit".into(),
    };
    let sentinel = IntegrateError::NumericalError {
        message: "callback sentinel".into(),
    };
    assert!(matches!(
        complete_integration::<()>(Err(sentinel), Some(callback)),
        Err(HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::WorkLimit,
            ..
        })
    ));
    let unexpected = IntegrateError::NumericalError {
        message: "invalid Simpson assembly".into(),
    };
    assert!(matches!(
        complete_integration::<()>(Err(unexpected), None),
        Err(HydraulicError::Numerical { .. })
    ));
    let callback = HydraulicError::Numerical {
        context: "retained callback arithmetic".into(),
    };
    assert!(matches!(
        complete_integration(Ok(()), Some(callback)),
        Err(HydraulicError::Numerical { .. })
    ));
}
