//! Side curves of fixed wet surveyed sources at exact target areas.
//!
//! A scalar endpoint trace lies on the wave curve of its interior cell. These rows give the
//! certified curve value `f_K(A)` and, where a compression is possible, the 1-shock magnitude.

use super::potential_range::{PotentialContext, PotentialEvaluation};
use super::wave_curve;
use crate::channel::{
    flow::surveyed::{
        PrismaticRiemannConfig,
        error::{HydraulicError, HydraulicResult, checked},
        work::WorkLedger,
    },
    section::{
        AreaAnchor, AreaDirection, AreaQuery, BoundaryAreas, GatheredBands, HydraulicGeometry,
        PhysicalEndpoint, PrismaticAreaPointRecovery, PrismaticSectionBatch,
    },
};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::{CertifiedExpansion, Interval},
    integrate::CertifiedSimpsonAlgorithms,
};

/// Recovered source points with the band and boundary rows of their section owners.
pub(crate) struct TraceCurveSources<R: Runtime<DType = DType>> {
    pub(crate) geometry: HydraulicGeometry<R>,
    pub(crate) bands: GatheredBands<R>,
    pub(crate) boundaries: BoundaryAreas<R>,
    pub(crate) source: PrismaticAreaPointRecovery<R>,
}

/// Wave-curve enclosures at exact target areas, one row per target.
pub(crate) struct TraceCurve<R: Runtime<DType = DType>> {
    /// Encloses `f_K(A)`: `Phi(A) - Phi(A_K)` below the source, the shock curve above it.
    pub(crate) curve: Interval<R>,
    /// Encloses the 1-shock magnitude `sqrt(A σ_p / A_K)` where `compression` holds one.
    pub(crate) shock: Interval<R>,
    /// One where the enclosures admit a target above the source area, zero otherwise.
    pub(crate) compression: Tensor<R>,
}

impl<R> TraceCurveSources<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>
        + CertifiedSimpsonAlgorithms<R>,
{
    /// Recovers each exact source area point on section owner zero.
    pub(crate) fn new(
        section: &PrismaticSectionBatch<R>,
        areas: &CertifiedExpansion<R>,
        config: &PrismaticRiemannConfig,
    ) -> HydraulicResult<Self> {
        let geometry = HydraulicGeometry::from_batch(section);
        let c = geometry.client();
        let owners = Tensor::<R>::zeros(areas.high().shape(), DType::I64, c.device())?;
        let source = section.recover_area_points_tensor(&owners, areas, &config.recovery)?;
        let (bands, boundaries) = checked(c, "surveyed trace source geometry", |a| {
            Ok((
                geometry.gather(a, &owners)?,
                geometry.boundary_areas(a)?.gather(a, &owners)?,
            ))
        })?;
        Ok(Self {
            geometry,
            bands,
            boundaries,
            source,
        })
    }

    /// Curve enclosures of source `rows[j]` at the exact area `areas[j]`.
    ///
    /// Every source must be wet. Each target must lie in `[0, cap area]`.
    pub(crate) fn curve(
        &self,
        rows: &Tensor<R>,
        areas: &Tensor<R>,
        config: &PrismaticRiemannConfig,
        panels: usize,
        work: &WorkLedger,
    ) -> HydraulicResult<TraceCurve<R>> {
        let c = self.geometry.client();
        let (bands, source, boundaries, query) =
            checked(c, "surveyed trace side-curve targets", |a| {
                let bands = self.bands.select(a, rows)?;
                let source = self.source.select(a, rows)?;
                let boundaries = self.boundaries.select(a, rows)?;
                let ones = Tensor::<R>::ones(areas.shape(), DType::F64, c.device())?;
                let scale = a.expansion_point(&ones)?;
                let parameter = a.point(areas)?;
                let query = AreaQuery::band_affine(
                    a,
                    &bands,
                    AreaAnchor::Zero,
                    AreaDirection::Add,
                    &scale,
                    &parameter,
                )?
                .prove_support(a, &bands)?;
                Ok((bands, source, boundaries, query))
            })?;
        let context = PotentialContext::new(
            &self.geometry,
            &bands,
            &source,
            &query,
            &boundaries,
            PotentialEvaluation {
                config,
                panels,
                work,
            },
        )?;
        let value = wave_curve::endpoint(&context, PhysicalEndpoint::Lower)?;
        checked(c, "surveyed trace side-curve values", |a| {
            let zero = Tensor::<R>::zeros(areas.shape(), DType::F64, c.device())?;
            if c.nonzero(&c.le(&value.curve.available, &zero)?)?.numel() != 0 {
                return Err(HydraulicError::Numerical {
                    context: "surveyed trace side curve is unavailable: supply a wet interior source and a supported target".into(),
                });
            }
            let actual = &value.actual_shock.value;
            let conditional = &value.conditional_shock.value;
            let actual_rows = c.gt(&actual.available, &zero)?;
            let conditional_rows = c.gt(&conditional.available, &zero)?;
            let pick = |actual: &Tensor<R>, conditional: &Tensor<R>| {
                c.where_cond(
                    &actual_rows,
                    actual,
                    &c.where_cond(&conditional_rows, conditional, &zero)?,
                )
            };
            let shock = a.bounds(
                &pick(actual.physical.lower(), conditional.physical.lower())?,
                &pick(actual.physical.upper(), conditional.physical.upper())?,
            )?;
            let curve = a.bounds(value.curve.physical.lower(), value.curve.physical.upper())?;
            Ok(TraceCurve {
                curve,
                shock,
                compression: c.maximum(&actual_rows, &conditional_rows)?,
            })
        })
    }
}
