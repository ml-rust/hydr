use super::{RunoffPoolBatch, RunoffPoolSetup, RunoffPoolWaterBalance};
use crate::storage::LevelPoolSources;
use hydr_structures::internal::bits;
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, CumulativeOps, IndexingOps, MidpointOps, ReduceOps,
    ScalarOps, ShapeOps, SortingOps, TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use solvr::interpolate::Interp1dAlgorithms;
use solvr::optimize::roots::{
    AndersonAlgorithms, BatchedBisectionAlgorithms, BatchedNewtonAlgorithms,
};

impl<R: Runtime<DType = DType>> RunoffPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + CumulativeOps<R>
        + IndexingOps<R>
        + MidpointOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>
        + AndersonAlgorithms<R>
        + BatchedNewtonAlgorithms<R>
        + BatchedBisectionAlgorithms<R>,
{
    pub(super) fn matches_setup(&self, other: &RunoffPoolSetup) -> bool {
        let own = self.setup();
        let a = &own.config;
        let b = &other.config;
        own.outlets == other.outlets
            && own.runoff_config.same_bits(&other.runoff_config)
            && own.subcatchments.len() == other.subcatchments.len()
            && own
                .subcatchments
                .iter()
                .zip(&other.subcatchments)
                .all(|(a, b)| a.id == b.id && a.parameters.same_bits(&b.parameters))
            && self.pools.matches_setup(&other.pools)
            && a.retry_limit == b.retry_limit
            && bits(
                &[
                    a.minimum_step_s,
                    a.maximum_step_s,
                    a.mass_absolute_m3,
                    a.mass_relative,
                ],
                &[
                    b.minimum_step_s,
                    b.maximum_step_s,
                    b.mass_absolute_m3,
                    b.mass_relative,
                ],
            )
    }
}
pub(super) fn same_sources(a: Option<&[LevelPoolSources]>, b: Option<&[LevelPoolSources]>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    bits(
                        &[
                            a.rain_rate_m_per_s,
                            a.potential_evaporation_rate_m_per_s,
                            a.bottom_loss_rate_m3_per_s,
                        ],
                        &[
                            b.rain_rate_m_per_s,
                            b.potential_evaporation_rate_m_per_s,
                            b.bottom_loss_rate_m3_per_s,
                        ],
                    )
                })
        }
        _ => false,
    }
}
pub(super) fn same_balance(a: &RunoffPoolWaterBalance, b: &RunoffPoolWaterBalance) -> bool {
    let values = |b: &RunoffPoolWaterBalance| {
        [
            b.storage_before_m3,
            b.storage_after_m3,
            b.rain_volume_m3,
            b.infiltration_volume_m3,
            b.evaporation_volume_m3,
            b.bottom_loss_volume_m3,
            b.soil_water_adjustment_volume_m3,
            b.internal_outlet_volume_m3,
            b.water_residual_m3,
            b.roundoff_bound_m3,
            b.physical_scale_m3,
        ]
    };
    bits(&values(a), &values(b))
}
