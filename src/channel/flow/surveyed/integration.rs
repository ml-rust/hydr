use super::band_lanes::IntegrationLanes;
use super::derivatives::{integrand, panel_fourth_bound};
use super::equation_ops::{choose, gravity_depth_root, scalar, scale};
use super::error::{HydraulicError, HydraulicResult, checked};
use super::work::{QueryWork, WorkLedger};
use super::{PrismaticRiemannConfig, PrismaticRiemannRejection};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticError, Interval};
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonValues, IntegrateError,
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
    let left = Tensor::zeros(&[lanes.band_index.numel()], DType::F64, c.device())?;
    let right = lanes.active.reshape(&[lanes.face_owner.numel()])?;
    let cause = Cell::new(None);
    let result = c.certified_simpson(
        &left,
        &right,
        CertifiedSimpsonOptions {
            panels,
            max_points: config.max_native_query_points,
        },
        |query| {
            let result = work
                .charge(QueryWork::SimpsonValue, query.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed Simpson values", |a| {
                        let x = a.bounds(&query.lower, &query.upper)?;
                        let value = integrand(a, &lanes, &query.owner, &x)?;
                        let point = a.point(&query.point)?;
                        let nominal_bounds = integrand(a, &lanes, &query.owner, &point)?;
                        let half_width = c.mul_scalar(
                            &c.sub(nominal_bounds.upper(), nominal_bounds.lower())?,
                            0.5,
                        )?;
                        let nominal = c.add(nominal_bounds.lower(), &half_width)?;
                        let nominal =
                            c.maximum(value.lower(), &c.minimum(&nominal, value.upper())?)?;
                        Ok(CertifiedSimpsonValues {
                            value: nominal,
                            lower: value.lower().clone(),
                            upper: value.upper().clone(),
                        })
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
        |panel| {
            let result = work
                .charge(QueryWork::SimpsonDerivative, panel.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed Simpson fourth derivatives", |a| {
                        let x = a.bounds(&panel.lower, &panel.upper)?;
                        panel_fourth_bound(a, &lanes, &panel.owner, &x)
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
    );
    let result = complete_integration(result, cause.take())?;
    checked(c, "surveyed potential assembly", |a| {
        let zero = scalar(a, 0.0)?;
        let one = scalar(a, 1.0)?;
        let two = scalar(a, 2.0)?;
        let pointed = bands.pointed.flatten()?;
        let pointed_factor = choose(a, &pointed, &two, &one)?;
        let root = gravity_depth_root(a, &lanes.gravity, &lanes.extent, Some(&pointed_factor))?;
        let bottom_exact = scale(a, &root, 2.0)?;
        let bottom_exact = choose(a, &lanes.bottom, &bottom_exact, &zero)?;
        let exact = choose(a, &lanes.analytic, &bottom_exact, &zero)?;
        let numerical = a.bounds(result.lower(), result.upper())?;
        let contributions = a.add(&exact, &numerical)?;
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

fn retain_cause(cause: &Cell<Option<HydraulicError>>, error: HydraulicError) -> IntegrateError {
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
