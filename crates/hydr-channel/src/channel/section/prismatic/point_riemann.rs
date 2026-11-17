use super::batch::PrismaticSectionBatch;
use crate::channel::flow::{
    surveyed::{
        PrismaticRiemannConfig, PrismaticRiemannError, error::into_public, work::WorkLedger,
    },
    surveyed_points::{PrismaticPointRiemannResult, point_riemann_entry},
};
use numr::ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::{
    dtype::DType,
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::{
    common::arithmetic::CertifiedExpansion,
    integrate::CertifiedSimpsonAlgorithms,
    optimize::{CertifiedMappedBisectionAlgorithms, CertifiedUpperBracketAlgorithms},
};

impl<R> PrismaticSectionBatch<R>
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
    /// Encloses horizontal surveyed-channel Godunov fluxes from exact conserved points.
    ///
    /// Owners use I64. Each area and discharge is a zero-radius F64 expansion with the owners' rank-one shape.
    /// Every operation runs on the configured native client. Inputs receive independent snapshots.
    /// Any face that stays unaccepted rejects the whole call and discards every partial result.
    pub fn riemann_flux_points_tensor(
        &self,
        owners: &Tensor<R>,
        left_area_m2: &CertifiedExpansion<R>,
        left_discharge_m3_per_s: &CertifiedExpansion<R>,
        right_area_m2: &CertifiedExpansion<R>,
        right_discharge_m3_per_s: &CertifiedExpansion<R>,
        config: &PrismaticRiemannConfig,
    ) -> Result<PrismaticPointRiemannResult<R>, PrismaticRiemannError> {
        let work = WorkLedger::new(config);
        point_riemann_entry::solve(
            self,
            owners,
            [
                left_area_m2,
                left_discharge_m3_per_s,
                right_area_m2,
                right_discharge_m3_per_s,
            ],
            config,
            &work,
        )
        .map_err(|error| into_public(error, work.snapshot()))
    }
}
