use super::{
    point_flux_association,
    point_flux_faces::{Continuation, Faces},
    point_flux_masks::equals,
    point_flux_types::{PointFluxKind, PointFluxStatus},
    point_quadrature_types::{PointQuadratureOutcome as Outcome, PointQuadratureProgress},
    point_root_session::PointRootSession,
};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PointRootSession<'_, '_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>
        + CertifiedMappedBisectionAlgorithms<R>
        + CertifiedUpperBracketAlgorithms<R>,
{
    /// Sharpens the star bracket of every accepted face that selected a star point, then
    /// reselects the faces whose bracket narrowed. The selected area then resolves the jump
    /// between the two sources, not only the depth tolerance.
    /// A face whose sharpening reaches a work ceiling reports `WorkLimit`.
    pub(super) fn sharpen_star_faces(
        &mut self,
        faces: &mut Faces<R>,
        count: usize,
    ) -> HydraulicResult<()> {
        let c = self.sources.geometry.client();
        let owners = checked::<R, _>(c, "point flux accepted star faces", |a| {
            faces.accepted_star(a)
        })?;
        if owners.numel() == 0 {
            return Ok(());
        }
        let progress = self.sharpen_wet_star(&owners)?;
        let improved = checked::<R, _>(c, "point flux sharpened star faces", |a| {
            let advanced = equals(a, &progress.outcome, Outcome::Advanced as i64)?;
            let narrowed = c.mul(&advanced, &progress.bracket_improved)?;
            Ok(c.index_select(
                &progress.invocation_owner,
                0,
                &c.nonzero(&narrowed)?.flatten()?,
            )?)
        })?;
        if improved.numel() != 0 {
            let reselected = self.select_point_fluxes(&improved)?;
            checked::<R, _>(c, "point flux sharpened reselection", |a| {
                faces.overwrite(a, count, reselected)
            })?;
        }
        checked::<R, _>(c, "point flux sharpening work ceiling", |a| {
            faces.record_sharpening(a, &progress)
        })
    }
}

impl<R> Faces<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Accepted face owners whose owning selection chose a star point.
    fn accepted_star(&self, a: &mut Arithmetic<'_, R>) -> HydraulicResult<Tensor<R>> {
        let c = a.client();
        let mut star = Tensor::zeros(self.owner.shape(), a.dtype(), c.device())?;
        for (index, selection) in self.selections.iter().enumerate() {
            let index = i64::try_from(index).map_err(|_| {
                super::point_root_session::invalid("point flux selection index exceeds i64")
            })?;
            for point in &selection.selected {
                let owners = &point.invocation_owner;
                if point.kind != PointFluxKind::Star || owners.numel() == 0 {
                    continue;
                }
                let positions = point_flux_association::positions(a, &self.association, owners)?;
                let current = equals(a, &c.index_select(&self.selection, 0, &positions)?, index)?;
                let previous = c.index_select(&star, 0, &positions)?;
                star = c.index_put(&star, 0, &positions, &c.maximum(&previous, &current)?)?;
            }
        }
        let accepted = equals(a, &self.status, PointFluxStatus::Accepted as i64)?;
        let open = c.mul(&accepted, &star)?;
        Ok(c.index_select(&self.owner, 0, &c.nonzero(&open)?.flatten()?)?)
    }

    /// Marks the faces whose sharpening reached a work ceiling as `WorkLimit`.
    fn record_sharpening(
        &mut self,
        a: &mut Arithmetic<'_, R>,
        progress: &PointQuadratureProgress<R>,
    ) -> HydraulicResult<()> {
        let c = a.client();
        let limited = equals(a, &progress.outcome, Outcome::WorkLimit as i64)?;
        let rows = c.nonzero(&limited)?.flatten()?;
        if rows.numel() == 0 {
            return Ok(());
        }
        let owners = c.index_select(&progress.invocation_owner, 0, &rows)?;
        let positions = point_flux_association::positions(a, &self.association, &owners)?;
        let index = Continuation::Star.index();
        let outcome = Tensor::full_scalar(
            owners.shape(),
            DType::I64,
            Outcome::WorkLimit as i64 as f64,
            c.device(),
        )?;
        self.outcome[index] = c.index_put(&self.outcome[index], 0, &positions, &outcome)?;
        let status = Tensor::full_scalar(
            owners.shape(),
            DType::I64,
            PointFluxStatus::WorkLimit as i64 as f64,
            c.device(),
        )?;
        self.status = c.index_put(&self.status, 0, &positions, &status)?;
        Ok(())
    }
}
