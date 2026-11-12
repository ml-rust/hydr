use std::cell::Cell;

use numr::{
    dtype::DType,
    ops::{CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedRootQuery, CertifiedRootValues},
};

use super::{
    curves::{sonic_residual, star_residual},
    error::HydraulicError,
    types::{Oracle, WaveSide},
    work::QueryWork,
};

pub(crate) fn evaluate<R>(
    oracle: &Oracle<'_, R>,
    face_owner: &Tensor<R>,
    query: &CertifiedRootQuery<R>,
    sonic: Option<WaveSide>,
    cause: &Cell<Option<HydraulicError>>,
) -> numr::error::Result<CertifiedRootValues<R>>
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
    let calculation = (|| {
        oracle
            .work
            .charge(QueryWork::Residual, query.point.numel())?;
        // Solvr owners index this invocation rather than the original face batch.
        let faces = oracle
            .geometry
            .client()
            .index_select(face_owner, 0, &query.owner)?;
        match sonic {
            Some(side) => sonic_residual(oracle, side, &faces, &query.point),
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
}
