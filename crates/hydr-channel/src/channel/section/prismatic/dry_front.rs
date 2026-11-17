//! Certified rarefaction-to-dry speeds of single surveyed states.

use super::{batch::PrismaticSectionBatch, hydraulic_geometry::HydraulicGeometry};
use crate::channel::flow::surveyed::{
    PrismaticRiemannConfig, PrismaticRiemannError,
    equation_ops::{choose, scalar},
    error::{HydraulicResult, checked, into_public},
    integration::potential,
    work::WorkLedger,
};
use numr::ops::{
    CompareOps, ConditionalOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps,
};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{common::arithmetic::CertifiedExpansion, integrate::CertifiedSimpsonAlgorithms};

impl<R> PrismaticSectionBatch<R>
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
    /// Certified `(lower of u - φ(A), upper of u + φ(A))` per state, `+0.0` where `A = 0`.
    ///
    /// `φ(A) = ∫_0^A c(a)/a da` is the dry-front Riemann invariant term. It increases with `A`.
    /// One certified potential at the upper area bound therefore bounds both directions.
    pub(crate) fn dry_front_bounds_tensor(
        &self,
        owners: &Tensor<R>,
        area_m2: &CertifiedExpansion<R>,
        discharge_m3_per_s: &CertifiedExpansion<R>,
        config: &PrismaticRiemannConfig,
    ) -> Result<(Tensor<R>, Tensor<R>), PrismaticRiemannError> {
        let work = WorkLedger::new(config);
        let run = || -> HydraulicResult<(Tensor<R>, Tensor<R>)> {
            config.check()?;
            let geometry = HydraulicGeometry::from_batch(self);
            let c = geometry.client();
            let (area, discharge) = checked(c, "surveyed dry-front state bounds", |a| {
                Ok((
                    a.expansion_interval(area_m2)?,
                    a.expansion_interval(discharge_m3_per_s)?,
                ))
            })?;
            let bands = checked(c, "surveyed dry-front gather", |a| {
                geometry.gather(a, owners)
            })?;
            let state = checked(c, "surveyed dry-front recovery", |a| {
                geometry.recover(a, &bands, area.upper(), &config.recovery)
            })?;
            let phi = potential(
                &geometry,
                &bands,
                &state,
                config,
                config.initial_simpson_panels,
                &work,
            )?;
            checked(c, "surveyed dry-front speeds", |a| {
                let zero = scalar(a, 0.0)?;
                let one = scalar(a, 1.0)?;
                let wet = c.gt(area.upper(), zero.upper())?;
                let denominator = choose(a, &wet, &area, &one)?;
                let velocity = a.div_positive_interval(&discharge, &denominator)?;
                let velocity = choose(a, &wet, &velocity, &zero)?;
                let rightward = a.add(&velocity, &phi)?;
                let leftward = a.sub(&velocity, &phi)?;
                let empty = Tensor::zeros(area.upper().shape(), DType::F64, c.device())?;
                Ok((
                    c.where_cond(&wet, leftward.lower(), &empty)?,
                    c.where_cond(&wet, rightward.upper(), &empty)?,
                ))
            })
        };
        run().map_err(|error| into_public(error, work.snapshot()))
    }
}
