use super::{
    paired_sources::PairedSources, potential_range::PotentialEvaluation, source_velocity,
    types::SourceVelocity,
};
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{HydraulicResult, checked},
        work::WorkLedger,
    },
    section::{
        BoundaryAreas, GatheredBands, HydraulicGeometry, PrismaticAreaPointRecovery,
        PrismaticSectionBatch,
    },
};
use numr::{
    dtype::DType,
    ops::{CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps, UtilityOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::CertifiedExpansion;

/// Captured conserved points with their recovered geometry, owned for one paired source graph.
pub(super) struct PointSourceParts<R: Runtime<DType = DType>> {
    pub(super) geometry: HydraulicGeometry<R>,
    pub(super) velocity: SourceVelocity<R>,
    pub(super) recovery: [PrismaticAreaPointRecovery<R>; 2],
    pub(super) bands: GatheredBands<R>,
    pub(super) boundaries: BoundaryAreas<R>,
}

impl<R> PointSourceParts<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Captures the four conserved points and recovers both source geometries.
    pub(super) fn capture(
        section: &PrismaticSectionBatch<R>,
        owners: &Tensor<R>,
        inputs: [&CertifiedExpansion<R>; 4],
        config: &PrismaticRiemannConfig,
    ) -> HydraulicResult<Self> {
        let geometry = HydraulicGeometry::from_batch(section);
        let captured =
            section.capture_riemann_points(owners, inputs[0], inputs[1], inputs[2], inputs[3])?;
        let velocity = source_velocity::prepare::<R>(geometry.client(), &captured)?;
        let owners = &captured.owners;
        let left =
            section.recover_area_points_tensor(owners, &captured.left_area, &config.recovery)?;
        let right =
            section.recover_area_points_tensor(owners, &captured.right_area, &config.recovery)?;
        let (bands, boundaries) =
            checked::<R, _>(geometry.client(), "point source geometry", |a| {
                Ok((
                    geometry.gather(a, owners)?,
                    geometry.boundary_areas(a)?.gather(a, owners)?,
                ))
            })?;
        Ok(Self {
            geometry,
            velocity,
            recovery: [left, right],
            bands,
            boundaries,
        })
    }

    /// Number of captured faces.
    pub(super) fn count(&self) -> usize {
        self.velocity.input().owners.numel()
    }

    /// I64 face positions 0..n in captured order.
    pub(super) fn sequential_rows(&self) -> HydraulicResult<Tensor<R>> {
        let c = self.geometry.client();
        Ok(c.arange(0., self.count() as f64, 1., DType::I64)?)
    }

    /// Binds one paired source graph at the initial Simpson panel count.
    /// `original_rows` names each face in rejections.
    pub(super) fn pair<'a>(
        &'a self,
        config: &'a PrismaticRiemannConfig,
        work: &'a WorkLedger,
        original_rows: &Tensor<R>,
    ) -> HydraulicResult<PairedSources<'a, R>> {
        PairedSources::new(
            &self.geometry,
            &self.bands,
            [&self.recovery[0], &self.recovery[1]],
            &self.velocity,
            &self.boundaries,
            PotentialEvaluation {
                config,
                panels: config.initial_simpson_panels,
                work,
            },
            original_rows,
        )
    }
}
