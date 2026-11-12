use std::cell::Cell;

use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ReduceOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{
        CertifiedUpperBracketAlgorithms, CertifiedUpperBracketOptions, CertifiedUpperBracketResult,
        CertifiedUpperBracketStatus,
    },
};

use super::{
    equation_ops::select_interval,
    error::{HydraulicError, HydraulicResult, PrismaticRiemannRejection, checked},
    root_calls::resolve_root,
    root_evaluation::evaluate,
    types::Oracle,
};

pub(crate) fn bracket_star<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
) -> HydraulicResult<CertifiedUpperBracketResult<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedUpperBracketAlgorithms<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    let c = oracle.geometry.client();
    let (lower, initial_upper, limit) = checked(c, "surveyed star bracket seed", |a| {
        let left = select_interval(a, &oracle.left.geometry.depth, face_owner)?;
        let right = select_interval(a, &oracle.right.geometry.depth, face_owner)?;
        let source_depth = a.bounds(
            &c.maximum(left.lower(), right.lower())?,
            &c.maximum(left.upper(), right.upper())?,
        )?;
        let bands = oracle.bands.select(a, face_owner)?;
        let normalized = a.div_positive_interval(&source_depth, &bands.cap_depth)?;
        let zero = Tensor::zeros(face_owner.shape(), DType::F64, c.device())?;
        let one = Tensor::ones(face_owner.shape(), DType::F64, c.device())?;
        let seed = c.minimum(&one, &c.maximum(&zero, normalized.upper())?)?;
        Ok((zero, seed, one))
    })?;
    oracle.work.begin_bracket()?;
    let cause = Cell::new(None);
    let result = c.bracket_upper_certified(
        |query| evaluate(oracle, face_owner, query, None, &cause),
        &lower,
        &initial_upper,
        &limit,
        &CertifiedUpperBracketOptions {
            max_expansions: oracle.config.max_bracket_expansions,
            max_points: oracle.config.root.max_points,
        },
    );
    let result = resolve_root(result, cause.take())?;
    oracle.work.finish_bracket(result.expansion_rounds())?;
    admit_brackets(oracle, face_owner, &result)?;
    Ok(result)
}

fn admit_brackets<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
    result: &CertifiedUpperBracketResult<R>,
) -> HydraulicResult<()>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = oracle.geometry.client();
    let bracketed = Tensor::from_slice(
        &[CertifiedUpperBracketStatus::Bracketed as i64],
        &[],
        c.device(),
    )?;
    let rejected = c.nonzero(&c.ne(result.status(), &bracketed)?)?.flatten()?;
    if rejected.numel() == 0 {
        return Ok(());
    }
    // Select the first original face before reading its status or section identity.
    let faces = c.index_select(face_owner, 0, &rejected)?;
    let first_face = c.min(&faces, &[0], true)?;
    let first_rows = c.nonzero(&c.eq(face_owner, &first_face)?)?.flatten()?;
    let first_row = c.min(&first_rows, &[0], true)?;
    let original_query = first_face.item::<i64>()?;
    let original_owner = c
        .index_select(&oracle.bands.section_owner, 0, &first_face)?
        .item::<i64>()?;
    let query = usize::try_from(original_query).map_err(|_| HydraulicError::Numerical {
        context: format!("Rejected query={original_query} cannot index the original input. Keep captured ownership immutable"),
    })?;
    let section_owner = usize::try_from(original_owner).map_err(|_| HydraulicError::Numerical {
        context: format!("Rejected section_owner={original_owner} cannot index the original sections. Keep captured ownership immutable"),
    })?;
    let code = c
        .index_select(result.status(), 0, &first_row)?
        .item::<i64>()?;
    let status = CertifiedUpperBracketStatus::from_code(code);
    let (reason, context) = match status {
        Some(CertifiedUpperBracketStatus::WorkLimit) => (
            PrismaticRiemannRejection::WorkLimit,
            "Wet star bracket reaches its expansion or point ceiling. Increase the bracket budget or split the face batch",
        ),
        Some(CertifiedUpperBracketStatus::NoRoot) => {
            let upper = c
                .index_select(result.upper(), 0, &first_row)?
                .item::<f64>()?;
            let residual_upper = c
                .index_select(&result.upper_residual().upper, 0, &first_row)?
                .item::<f64>()?;
            if upper == 1.0 && residual_upper < 0.0 {
                (
                    PrismaticRiemannRejection::CapExceeded,
                    "Wet star residual remains negative at the physical surveyed cap. Extend surveyed support or change the input state",
                )
            } else {
                (
                    PrismaticRiemannRejection::RootBracketUnresolved,
                    "Wet star lower residual has an incompatible positive sign. Refine potential enclosures and review wet-star classification",
                )
            }
        }
        Some(CertifiedUpperBracketStatus::Unresolved) => (
            PrismaticRiemannRejection::RootBracketUnresolved,
            "Wet star endpoint signs do not certify existence at the physical cap. Refine potential enclosures",
        ),
        _ => {
            return Err(HydraulicError::Numerical {
                context: format!(
                    "Certified star bracket returns unexpected status={code}. Review native bracket status storage"
                ),
            });
        }
    };
    Err(HydraulicError::Rejected {
        reason,
        query: Some(query),
        section_owner: Some(section_owner),
        context: context.into(),
    })
}
