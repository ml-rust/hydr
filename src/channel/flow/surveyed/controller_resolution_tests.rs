use super::{
    controller::resolve_acceptance,
    error::{HydraulicError, PrismaticRiemannRejection},
    root_calls::resolve_root,
};
use solvr::{common::arithmetic::ArithmeticError, optimize::OptimizeError};

fn rejected(reason: PrismaticRiemannRejection) -> HydraulicError {
    HydraulicError::Rejected {
        reason,
        query: Some(3),
        section_owner: Some(7),
        context: "retained callback cause".into(),
    }
}

#[test]
fn root_completion_error_precedes_retained_callback_cause() {
    let result: Result<(), _> = Err(OptimizeError::NumericalError {
        message: "sticky arithmetic completion".into(),
    });
    let error = resolve_root(result, Some(rejected(PrismaticRiemannRejection::WorkLimit)));
    assert!(
        matches!(error, Err(HydraulicError::Numerical { context }) if context.contains("sticky arithmetic completion"))
    );
}

#[test]
fn callback_wrapper_restores_typed_cause_and_identities() {
    let result: Result<(), _> = Err(OptimizeError::NumrError("callback wrapper".into()));
    let error = resolve_root(
        result,
        Some(rejected(PrismaticRiemannRejection::QuadratureUnresolved)),
    );
    assert!(matches!(error, Err(HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::QuadratureUnresolved,
        query: Some(3), section_owner: Some(7), context,
    }) if context == "retained callback cause"));
}

#[test]
fn backend_error_without_callback_cause_remains_runtime() {
    let result: Result<(), _> = Err(OptimizeError::NumrError("native backend error".into()));
    assert!(matches!(
        resolve_root(result, None),
        Err(HydraulicError::Runtime(_))
    ));
}

#[test]
fn root_admission_point_ceiling_remains_work_limit() {
    let result: Result<(), _> = Err(OptimizeError::InvalidParameter {
        parameter: "max_points".into(),
        message: "endpoint admission exceeds max_points=1".into(),
    });
    assert!(
        matches!(resolve_root(result, None), Err(HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::WorkLimit, context, ..
    }) if context.contains("max_points=1"))
    );
}

#[test]
fn retained_callback_cause_never_allows_solver_success() {
    let result = resolve_root(Ok(()), Some(rejected(PrismaticRiemannRejection::WorkLimit)));
    assert!(matches!(result, Err(HydraulicError::Numerical { .. })));
}

#[test]
fn root_work_limit_remaps_only_flux_width_rejection() {
    let result = resolve_acceptance(
        rejected(PrismaticRiemannRejection::FluxBudgetExceeded),
        true,
    );
    assert!(matches!(result, HydraulicError::Rejected {
        reason: PrismaticRiemannRejection::WorkLimit, context, ..
    } if context.contains("FluxBudgetExceeded")));
    let result = resolve_acceptance(
        rejected(PrismaticRiemannRejection::FluxBudgetExceeded),
        false,
    );
    assert!(matches!(
        result,
        HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::FluxBudgetExceeded,
            ..
        }
    ));
}

#[test]
fn root_work_limit_preserves_fatal_acceptance_errors() {
    let arithmetic = resolve_acceptance(
        HydraulicError::Arithmetic(ArithmeticError::Range {
            stage: "acceptance",
        }),
        true,
    );
    assert!(matches!(
        arithmetic,
        HydraulicError::Arithmetic(ArithmeticError::Range {
            stage: "acceptance"
        })
    ));
    let runtime = resolve_acceptance(
        HydraulicError::Runtime(numr::error::Error::InvalidArgument {
            arg: "acceptance",
            reason: "backend rejection".into(),
        }),
        true,
    );
    assert!(matches!(runtime, HydraulicError::Runtime(_)));
    let numerical = resolve_acceptance(
        HydraulicError::Numerical {
            context: "sticky completion".into(),
        },
        true,
    );
    assert!(
        matches!(numerical, HydraulicError::Numerical { context } if context == "sticky completion")
    );
    let geometry = resolve_acceptance(
        rejected(PrismaticRiemannRejection::GeometryUnresolved),
        true,
    );
    assert!(matches!(
        geometry,
        HydraulicError::Rejected {
            reason: PrismaticRiemannRejection::GeometryUnresolved,
            query: Some(3),
            section_owner: Some(7),
            ..
        }
    ));
}
