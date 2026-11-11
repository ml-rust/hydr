use std::cell::Cell;

use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, TensorOps, TwoSumOps,
        TypeConversionOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedBisectionAlgorithms, CertifiedRootValues, OptimizeError},
};

use super::{
    curves::{sonic_residual, star_residual},
    error::{HydraulicError, HydraulicResult, checked},
    types::{Oracle, WaveSide},
    work::QueryWork,
};

pub(crate) struct RootSet<R: Runtime<DType = DType>> {
    pub(crate) face_owner: Tensor<R>,
    pub(crate) normalized_depth: solvr::common::arithmetic::Interval<R>,
    pub(crate) status: Tensor<R>,
}

pub(crate) fn solve_star<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
) -> HydraulicResult<RootSet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = oracle.geometry.client();
    let lower = Tensor::zeros(face_owner.shape(), DType::F64, c.device())?;
    let upper = Tensor::ones(face_owner.shape(), DType::F64, c.device())?;
    solve(oracle, face_owner, &lower, &upper, None)
}

pub(crate) fn solve_sonic<R>(
    oracle: &Oracle<'_, R>,
    side: WaveSide,
    face_owner: &Tensor<R>,
    lower_t: &Tensor<R>,
    upper_t: &Tensor<R>,
) -> HydraulicResult<RootSet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    solve(oracle, face_owner, lower_t, upper_t, Some(side))
}

fn solve<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
    lower: &Tensor<R>,
    upper: &Tensor<R>,
    sonic: Option<WaveSide>,
) -> HydraulicResult<RootSet<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedBisectionAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = oracle.geometry.client();
    let admission_points =
        face_owner
            .numel()
            .checked_mul(2)
            .ok_or_else(|| HydraulicError::Rejected {
                reason: super::error::PrismaticRiemannRejection::WorkLimit,
                query: None,
                section_owner: None,
                context: "Root admission point count overflows usize. Split the face batch".into(),
            })?;
    if admission_points > oracle.config.root.max_points {
        return Err(HydraulicError::Rejected {
            reason: super::error::PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: format!(
                "Root admission needs {admission_points} points, root.max_points={}. Increase the root budget or split the face batch",
                oracle.config.root.max_points
            ),
        });
    }
    let derivative = checked(c, "surveyed root derivative", |a| {
        let bands = oracle.bands.select(a, face_owner)?;
        let gravity = a.point(&Tensor::from_slice(
            &[oracle.config.gravity_m_per_s2],
            &[],
            c.device(),
        )?)?;
        let ratio = a.div_positive_interval(&bands.cap_width, &bands.cap_area)?;
        let product = a.mul(&gravity, &ratio)?;
        let slope = a.sqrt_nonnegative(&product)?;
        let normalized = a.mul(&bands.cap_depth, &slope)?;
        let multiplier = if sonic.is_some() { 1.0 } else { 2.0 };
        let factor = a.point(&Tensor::from_slice(&[multiplier], &[], c.device())?)?;
        let bound = a.mul(&normalized, &factor)?;
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        if face_owner.numel() != 0
            && c.cast(
                &c.any(&c.le(bound.lower(), &zero)?, &[], false)?,
                DType::F64,
            )?
            .item::<f64>()?
                != 0.0
        {
            return Err(HydraulicError::Rejected { reason: super::error::PrismaticRiemannRejection::GeometryUnresolved, query: None, section_owner: None, context: "Normalized root derivative has a nonpositive lower bound. Review cap conditioning".into() });
        }
        Ok(bound.lower().clone())
    })?;
    oracle.work.begin_root()?;
    let cause = Cell::new(None);
    let result = c.bisect_certified(
        |query| {
            let calculation = (|| {
                oracle
                    .work
                    .charge(QueryWork::Residual, query.point.numel())?;
                // Solvr owners index this invocation, rather than the original face batch.
                let faces = c.index_select(face_owner, 0, &query.owner)?;
                match sonic.as_ref() {
                    Some(WaveSide::Left) => {
                        sonic_residual(oracle, WaveSide::Left, &faces, &query.point)
                    }
                    Some(WaveSide::Right) => {
                        sonic_residual(oracle, WaveSide::Right, &faces, &query.point)
                    }
                    None => star_residual(oracle, &faces, &query.point),
                }
            })();
            match calculation {
                Ok(value) => Ok(CertifiedRootValues {
                    lower: value.lower().clone(),
                    upper: value.upper().clone(),
                }),
                Err(error) => {
                    let first = cause.take();
                    cause.set(Some(first.unwrap_or(error)));
                    Err(numr::error::Error::InvalidArgument {
                        arg: "surveyed residual",
                        reason: "Hydraulic callback rejected. Read the retained typed cause".into(),
                    })
                }
            }
        },
        lower,
        upper,
        &derivative,
        &oracle.config.root,
    );
    let result = resolve_root(result, cause.take())?;
    oracle.work.finish_root(result.iterations())?;
    checked(c, "surveyed certified roots", |a| {
        Ok(RootSet {
            face_owner: c.index_select(face_owner, 0, result.root_owner())?,
            normalized_depth: a.bounds(result.root_lower(), result.root_upper())?,
            status: result.status().clone(),
        })
    })
}

pub(crate) fn resolve_root<T>(
    result: Result<T, OptimizeError>,
    cause: Option<HydraulicError>,
) -> HydraulicResult<T> {
    match result {
        Ok(value) => match cause {
            None => Ok(value),
            Some(_) => Err(HydraulicError::Numerical { context: "Certified root returned success after a retained callback rejection. Discard the result and review callback completion".into() }),
        },
        Err(OptimizeError::NumrError(message)) => match cause {
            Some(error) => Err(error),
            None => Err(HydraulicError::Runtime(numr::error::Error::InvalidArgument {
                arg: "certified surveyed root",
                reason: message,
            })),
        },
        Err(error @ OptimizeError::NumericalError { .. }) => Err(HydraulicError::Numerical { context: error.to_string() }),
        Err(error) if matches!(&error, OptimizeError::InvalidParameter { parameter, .. } if parameter == "max_points") => Err(HydraulicError::Rejected {
            reason: super::error::PrismaticRiemannRejection::WorkLimit,
            query: None,
            section_owner: None,
            context: error.to_string(),
        }),
        Err(error @ (OptimizeError::InvalidParameter { .. } | OptimizeError::InvalidInput { .. })) => Err(HydraulicError::Geometry(crate::channel::section::CrossSectionError::InvalidInput {
            section: None,
            context: error.to_string(),
        })),
        Err(error) => Err(HydraulicError::Numerical { context: error.to_string() }),
    }
}
