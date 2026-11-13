use super::{
    dry_vacuum_types::{
        DryVacuumCandidate, DryVacuumDry, DryVacuumExterior, DryVacuumPendingFan,
        DryVacuumPossibilities,
    },
    point_root_session::PointRootSession,
    source_sonic_types::{SonicSide, SourceSonicCertificate, SourceSonicStatus},
};
use crate::channel::{
    flow::surveyed::error::{HydraulicResult, checked},
    section::{AreaAnchor, AreaDirection, AreaQuery},
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, IndexingOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};

pub(super) fn assemble<R>(
    session: &PointRootSession<'_, '_, R>,
    owners: &Tensor<R>,
    parent: &Tensor<R>,
    possible: &DryVacuumPossibilities<R>,
    sonic: Vec<SourceSonicCertificate<R>>,
) -> HydraulicResult<(Vec<DryVacuumCandidate<R>>, Tensor<R>)>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    let c = session.sources.geometry.client();
    checked::<R, _>(c, "dry vacuum physical candidate coverage", |a| {
        let zero = Tensor::zeros(owners.shape(), a.dtype(), c.device())?;
        let one = Tensor::full_scalar(owners.shape(), a.dtype(), 1., c.device())?;
        let fullzero = Tensor::zeros(session.invocation_rows.shape(), a.dtype(), c.device())?;
        let mut physical = [fullzero.clone(), fullzero.clone()];
        let mut present = [fullzero.clone(), fullzero];
        let mut output = Vec::new();
        for certificate in sonic {
            let side = certificate.side.index();
            let success = c.eq(
                &certificate.status,
                &Tensor::zeros(&[], DType::I64, c.device())?,
            )?;
            let success = c.where_cond(
                &success,
                &Tensor::full_scalar(certificate.status.shape(), a.dtype(), 1., c.device())?,
                &Tensor::zeros(certificate.status.shape(), a.dtype(), c.device())?,
            )?;
            physical[side] =
                c.index_put(&physical[side], 0, &certificate.invocation_owner, &success)?;
            present[side] = c.index_put(
                &present[side],
                0,
                &certificate.invocation_owner,
                &Tensor::full_scalar(certificate.status.shape(), a.dtype(), 1., c.device())?,
            )?;
            output.push(DryVacuumCandidate::Fan(Box::new(certificate)));
        }
        let mut complete = one.clone();
        let mut dry = possible.dry.clone();
        for side in [SonicSide::Left, SonicSide::Right] {
            let i = side.index();
            let status = c.index_select(&session.sonic.status[i], 0, owners)?;
            let is = |kind: SourceSonicStatus| -> HydraulicResult<Tensor<R>> {
                Ok(c.where_cond(
                    &c.eq(
                        &status,
                        &Tensor::full_scalar(&[], DType::I64, kind as i64 as f64, c.device())?,
                    )?,
                    &one,
                    &zero,
                )?)
            };
            let fan = if i == 0 {
                &possible.left_fan
            } else {
                &possible.right_fan
            };
            let source = c.mul(fan, &is(SourceSonicStatus::SourceContact)?)?;
            let drycontact = c.mul(fan, &is(SourceSonicStatus::DryContact)?)?;
            dry = c.maximum(&dry, &drycontact)?;
            let exterior = if i == 0 {
                &possible.left_exterior
            } else {
                &possible.right_exterior
            };
            let exterior = c.maximum(exterior, &source)?;
            let rows = c.nonzero(&exterior)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() != 0 {
                let selected_parent = c.index_select(parent, 0, &rows)?;
                let selected = session.sources.select_sources(a, &selected_parent)?;
                let query = super::potential_range::source_query(
                    session.sources.geometry,
                    &selected.bands,
                    &selected.sources[i],
                    &selected.original_rows,
                )?;
                let input = selected.velocity.input();
                let area = if i == 0 {
                    &input.left_area
                } else {
                    &input.right_area
                };
                let discharge = if i == 0 {
                    &input.left_discharge
                } else {
                    &input.right_discharge
                };
                let velocity = if i == 0 {
                    &selected.velocity.left
                } else {
                    &selected.velocity.right
                };
                let area = a.expansion_bounds(area.high(), area.low(), area.abs_error())?;
                let discharge =
                    a.expansion_bounds(discharge.high(), discharge.low(), discharge.abs_error())?;
                let velocity =
                    a.expansion_bounds(velocity.high(), velocity.low(), velocity.abs_error())?;
                let [left, right] = selected.sources;
                let recovery = if i == 0 { left } else { right };
                output.push(DryVacuumCandidate::Exterior(Box::new(DryVacuumExterior {
                    invocation_owner: c.index_select(owners, 0, &rows)?,
                    parent_rows: selected_parent,
                    side,
                    query,
                    area,
                    discharge,
                    velocity,
                    recovery,
                })));
            }
            let valid = c.maximum(
                &source,
                &c.maximum(&drycontact, &c.index_select(&physical[i], 0, owners)?)?,
            )?;
            complete = c.mul(&complete, &c.where_cond(fan, &valid, &one)?)?;
            let known = c.maximum(
                &source,
                &c.maximum(&drycontact, &c.index_select(&present[i], 0, owners)?)?,
            )?;
            let pending = c.mul(fan, &c.sub(&one, &known)?)?;
            let rows = c.nonzero(&pending)?;
            let rows = rows.reshape(&[rows.numel()])?;
            if rows.numel() != 0 {
                output.push(DryVacuumCandidate::PendingFan(Box::new(
                    DryVacuumPendingFan {
                        invocation_owner: c.index_select(owners, 0, &rows)?,
                        parent_rows: c.index_select(parent, 0, &rows)?,
                        side,
                        source_status: c.index_select(&status, 0, &rows)?,
                    },
                )));
            }
        }
        let rows = c.nonzero(&dry)?;
        let rows = rows.reshape(&[rows.numel()])?;
        if rows.numel() != 0 {
            let selected_parent = c.index_select(parent, 0, &rows)?;
            let selected = session.sources.select_sources(a, &selected_parent)?;
            let z = Tensor::zeros(rows.shape(), a.dtype(), c.device())?;
            let area = a.expansion_point(&z)?;
            let parameter = a.point(&z)?;
            let query = AreaQuery::band_affine(
                a,
                &selected.bands,
                AreaAnchor::Zero,
                AreaDirection::Add,
                &area,
                &parameter,
            )?;
            c.copy_into(&query.original_rows, &selected.original_rows)?;
            let query = query.prove_support(a, &selected.bands)?;
            output.push(DryVacuumCandidate::Dry(Box::new(DryVacuumDry {
                invocation_owner: c.index_select(owners, 0, &rows)?,
                parent_rows: selected_parent,
                query,
                area,
                discharge: a.expansion_point(&z)?,
                depth: a.point(&z)?,
            })));
        }
        let coverage = c.maximum(
            &possible.left_exterior,
            &c.maximum(
                &possible.left_fan,
                &c.maximum(
                    &possible.dry,
                    &c.maximum(&possible.right_fan, &possible.right_exterior)?,
                )?,
            )?,
        )?;
        Ok((output, c.mul(&complete, &coverage)?))
    })
}
