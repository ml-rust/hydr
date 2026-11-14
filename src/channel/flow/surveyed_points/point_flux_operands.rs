use super::{
    point_candidate_types::{PointCandidate, PointCandidateCertificates},
    point_flux_types::PointFluxKind,
};
use crate::channel::{
    flow::surveyed::{
        equation_ops::select_interval,
        error::{Arithmetic, HydraulicResult},
    },
    section::SupportedAreaQuery,
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps,
        TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(super) struct Operands<R: Runtime<DType = DType>> {
    pub(super) candidate: usize,
    pub(super) kind: PointFluxKind,
    pub(super) owners: Tensor<R>,
    pub(super) parent: Tensor<R>,
    pub(super) query: SupportedAreaQuery<R>,
    pub(super) discharge: Interval<R>,
    pub(super) captured: Option<[CertifiedExpansion<R>; 2]>,
}

pub(super) fn candidate<R>(
    a: &mut Arithmetic<'_, R>,
    result: &PointCandidateCertificates<R>,
    index: usize,
) -> HydraulicResult<Option<Operands<R>>>
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
    let Some(candidate) = result.candidates.get(index) else {
        return Err(super::point_root_session::invalid(
            "point flux candidate index is absent",
        ));
    };
    let (kind, owners, parent, query, rows, discharge, captured) = match candidate {
        PointCandidate::Exterior(e) => {
            let rows = c.arange(0., e.invocation_owner.numel() as f64, 1., DType::I64)?;
            let q = a.expansion_interval(&e.discharge)?;
            (
                PointFluxKind::Exterior(e.side),
                &e.invocation_owner,
                &e.parent_rows,
                &e.query,
                rows,
                q,
                Some([
                    copy_expansion(a, &e.area)?,
                    copy_expansion(a, &e.discharge)?,
                ]),
            )
        }
        PointCandidate::Dry(d) => {
            let rows = c.arange(0., d.invocation_owner.numel() as f64, 1., DType::I64)?;
            let q = a.expansion_interval(&d.discharge)?;
            (
                PointFluxKind::Dry,
                &d.invocation_owner,
                &d.parent_rows,
                &d.query,
                rows,
                q,
                Some([
                    copy_expansion(a, &d.area)?,
                    copy_expansion(a, &d.discharge)?,
                ]),
            )
        }
        PointCandidate::Star(s) => {
            let wave = result.waves.get(s.wave).ok_or_else(|| {
                super::point_root_session::invalid("point flux star wave evidence is absent")
            })?;
            let q = select_interval(a, &wave.star.discharge, &s.rows)?;
            return Ok(Some(Operands {
                candidate: index,
                kind: PointFluxKind::Star,
                owners: copy_tensor(a, &s.invocation_owner)?,
                parent: copy_tensor(a, &s.parent_rows)?,
                query: wave.star.query.select(a, &s.rows)?,
                discharge: q,
                captured: None,
            }));
        }
        PointCandidate::Fan(f) => {
            let cert = &f.certificate;
            let rows = c.nonzero(&f.active)?.flatten()?;
            let q = select_interval(a, &cert.discharge, &rows)?;
            (
                PointFluxKind::Fan(cert.side),
                &cert.invocation_owner,
                &cert.parent_rows,
                &cert.query,
                rows,
                q,
                None,
            )
        }
        PointCandidate::Pending(_) => return Ok(None),
    };
    if rows.numel() == 0 {
        return Ok(None);
    }
    Ok(Some(Operands {
        candidate: index,
        kind,
        owners: c.index_select(owners, 0, &rows)?,
        parent: c.index_select(parent, 0, &rows)?,
        query: query.select(a, &rows)?,
        discharge,
        captured,
    }))
}

pub(super) fn select<R>(
    a: &mut Arithmetic<'_, R>,
    operand: &Operands<R>,
    rows: &Tensor<R>,
) -> HydraulicResult<Operands<R>>
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
    let captured = if let Some([area, q]) = &operand.captured {
        Some([
            super::potential_range::select_expansion(a, area, rows)?,
            super::potential_range::select_expansion(a, q, rows)?,
        ])
    } else {
        None
    };
    Ok(Operands {
        candidate: operand.candidate,
        kind: operand.kind,
        owners: c.index_select(&operand.owners, 0, rows)?,
        parent: c.index_select(&operand.parent, 0, rows)?,
        query: operand.query.select(a, rows)?,
        discharge: select_interval(a, &operand.discharge, rows)?,
        captured,
    })
}

pub(super) fn copy_tensor<R>(a: &Arithmetic<'_, R>, value: &Tensor<R>) -> HydraulicResult<Tensor<R>>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let owned = Tensor::empty(value.shape(), value.dtype(), a.client().device())?;
    a.client().copy_into(&owned, value)?;
    Ok(owned)
}

fn copy_expansion<R>(
    a: &mut Arithmetic<'_, R>,
    value: &CertifiedExpansion<R>,
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
    Ok(a.expansion_bounds(
        &copy_tensor(a, value.high())?,
        &copy_tensor(a, value.low())?,
        &copy_tensor(a, value.abs_error())?,
    )?)
}
