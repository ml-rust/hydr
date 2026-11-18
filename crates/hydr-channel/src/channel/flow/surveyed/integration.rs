use super::band_lanes::IntegrationLanes;
use super::derivatives::{integrand, panel_fourth_bound};
use super::equation_ops::{choose, gravity_depth_root, scalar, scale};
use super::error::{Arithmetic, HydraulicError, HydraulicResult, checked};
use super::lane_quadrature::numerical_contributions;
use super::work::WorkLedger;
use super::{PrismaticRiemannConfig, PrismaticRiemannRejection};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticError, Interval};
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonResult, IntegrateError,
};
use std::cell::Cell;

pub(crate) fn potential<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    state: &Constitutive<R>,
    config: &PrismaticRiemannConfig,
    panels: usize,
    work: &WorkLedger,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = geometry.client();
    let lanes = checked(c, "surveyed integration lanes", |a| {
        IntegrationLanes::build(a, bands, state, config.gravity_m_per_s2)
    })?;
    integrate_lanes(c, &lanes, &bands.pointed, config, panels, work)
}

pub(crate) fn integrate_lanes<R>(
    c: &R::Client,
    lanes: &IntegrationLanes<R>,
    pointed: &Tensor<R>,
    config: &PrismaticRiemannConfig,
    panels: usize,
    work: &WorkLedger,
) -> HydraulicResult<Interval<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let left = Tensor::zeros(&[lanes.band_index.numel()], DType::F64, c.device())?;
    let right = lanes.active.reshape(&[lanes.face_owner.numel()])?;
    let result = numerical_contributions(
        c,
        &left,
        &right,
        CertifiedSimpsonOptions {
            panels,
            max_points: config.max_native_query_points,
        },
        work,
        |a, owner, x| integrand(a, lanes, owner, x),
        |a, owner, x| panel_fourth_bound(a, lanes, owner, x),
    )?;
    checked(c, "surveyed potential assembly", |a| {
        let contributions = assemble_lane_contributions(a, lanes, pointed, &result)?;
        let shape = [lanes.face_count, lanes.band_count];
        let lower = contributions.lower().reshape(&shape)?;
        let upper = contributions.upper().reshape(&shape)?;
        let empty = Tensor::zeros(&[lanes.face_count], DType::F64, c.device())?;
        let mut total = a.point(&empty)?;
        for band in 0..lanes.band_count {
            let term = a.bounds(
                &lower.narrow(1, band, 1)?.squeeze(Some(1)),
                &upper.narrow(1, band, 1)?.squeeze(Some(1)),
            )?;
            total = a.add(&total, &term)?;
        }
        Ok(total)
    })
}

pub(crate) fn assemble_lane_contributions<R>(
    a: &mut Arithmetic<'_, R>,
    lanes: &IntegrationLanes<R>,
    pointed: &Tensor<R>,
    result: &CertifiedSimpsonResult<R>,
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
    let zero = scalar(a, 0.0)?;
    let one = scalar(a, 1.0)?;
    let two = scalar(a, 2.0)?;
    let pointed = pointed.flatten()?;
    let pointed_factor = choose(a, &pointed, &two, &one)?;
    let root = gravity_depth_root(a, &lanes.gravity, &lanes.extent, Some(&pointed_factor))?;
    let bottom_exact = scale(a, &root, 2.0)?;
    let bottom_exact = choose(a, &lanes.bottom, &bottom_exact, &zero)?;
    let exact = choose(a, &lanes.analytic, &bottom_exact, &zero)?;
    let numerical = a.bounds(result.lower(), result.upper())?;
    Ok(a.add(&exact, &numerical)?)
}

pub(super) fn retain_cause(
    cause: &Cell<Option<HydraulicError>>,
    error: HydraulicError,
) -> IntegrateError {
    let first = cause.take();
    cause.set(Some(first.unwrap_or(error)));
    IntegrateError::NumericalError {
        message: "Surveyed hydraulic callback rejected its query. Inspect the retained typed cause"
            .into(),
    }
}

pub(super) fn integration_error(error: IntegrateError) -> HydraulicError {
    match error {
        IntegrateError::UnsupportedArithmetic {
            dtype,
            failed_probes,
        } => ArithmeticError::UnsupportedArithmetic {
            dtype,
            failed_probes,
        }
        .into(),
        IntegrateError::ArithmeticRange { stage } => ArithmeticError::Range { stage }.into(),
        IntegrateError::NumrError(context) => {
            HydraulicError::Runtime(numr::error::Error::Backend(context))
        }
        IntegrateError::NumericalError { message } => HydraulicError::Numerical {
            context: format!(
                "Surveyed Simpson numerical arithmetic rejected its query: {message}. Check the callback bounds"
            ),
        },
        IntegrateError::InvalidParameter { parameter, message } if parameter == "max_points" => {
            HydraulicError::Rejected {
                reason: PrismaticRiemannRejection::WorkLimit,
                query: None,
                section_owner: None,
                context: format!("Surveyed Simpson parameter={parameter}: {message}"),
            }
        }
        error => HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::QuadratureUnresolved,
            query: None,
            section_owner: None,
            context: format!(
                "Surveyed Simpson returned {error}. Inspect geometry conditioning and the frozen panel ceiling"
            ),
        },
    }
}

pub(super) fn complete_integration<T>(
    result: Result<T, IntegrateError>,
    cause: Option<HydraulicError>,
) -> HydraulicResult<T> {
    match (result, cause) {
        (Err(error @ (IntegrateError::ArithmeticRange { .. } | IntegrateError::UnsupportedArithmetic { .. } | IntegrateError::NumrError(_))), _) => Err(integration_error(error)),
        (Err(IntegrateError::NumericalError { .. }), Some(error)) => Err(error),
        (Ok(_), Some(_)) => Err(HydraulicError::Numerical {
            context: "Surveyed Simpson returned success after its callback rejected a query. Rebuild the complete integral".into(),
        }),
        (Err(error), _) => Err(integration_error(error)),
        (Ok(value), None) => Ok(value),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simpson_runtime_errors_retain_their_diagnostics() {
        let diagnostic = "configured CUDA client rejected allocation";
        let mapped = integration_error(IntegrateError::NumrError(diagnostic.into()));
        assert!(
            matches!(mapped, HydraulicError::Runtime(numr::error::Error::Backend(context)) if context == diagnostic)
        );
    }

    #[test]
    fn simpson_query_budget_errors_retain_the_work_limit() {
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
}
