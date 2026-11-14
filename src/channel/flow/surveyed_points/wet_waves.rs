use super::{
    wet_depth_types::WetDepthSession,
    wet_star_types::WetStarStatus,
    wet_wave_types::{WetWaveDomain, WetWaveStatus},
    wet_waves_endpoint, wet_waves_regions, wet_waves_side,
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        BinaryOps, CompareOps, ConditionalOps, MidpointOps, NextRepresentableOps, ScalarOps,
        TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::integrate::CertifiedSimpsonAlgorithms;

impl<R> WetDepthSession<'_, '_, R>
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
    /// Retain possible wet-wave positions without selecting a trace or accepting a stationary discontinuity.
    pub(super) fn certify_wet_waves(&mut self) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        self.sources
            .check_root_witness(self.invocation.source_witness())?;
        if self.wet.terminal {
            return Err(super::point_root_session::invalid(
                "wet certificates remain sealed after an earlier numerical error",
            ));
        }
        self.check_domains()?;
        let stars = self.certify_wet_star()?;
        self.wet_waves_from(stars)
    }

    pub(super) fn certify_wet_waves_selected(
        &mut self,
        demand: &Tensor<R>,
    ) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        let stars = self.certify_wet_star_selected(demand)?;
        self.wet_waves_from(stars)
    }

    fn wet_waves_from(
        &mut self,
        stars: Vec<super::wet_star_types::WetStarDomain<R>>,
    ) -> HydraulicResult<Vec<WetWaveDomain<R>>> {
        let c = self.sources.geometry.client();
        let mut result = Vec::with_capacity(stars.len());
        for star in stars {
            let domain = checked(c, "wet wave positions over established root domain", |a| {
                let selected = self.sources.select_rows(&star.parent_rows, &star.query)?;
                let endpoint_sound = wet_waves_endpoint::sounds(a, self.sources, &selected, &star)?;
                let left = wet_waves_side::evaluate(
                    a,
                    &selected,
                    &star,
                    &endpoint_sound,
                    self.sources.evaluation.config.gravity_m_per_s2,
                    0,
                )?;
                let right = wet_waves_side::evaluate(
                    a,
                    &selected,
                    &star,
                    &endpoint_sound,
                    self.sources.evaluation.config.gravity_m_per_s2,
                    1,
                )?;
                let sides = [left, right];
                let mut regions = wet_waves_regions::evaluate(a, &sides)?;
                let covered = c.maximum(
                    &regions.left_exterior,
                    &c.maximum(
                        &regions.left_fan,
                        &c.maximum(
                            &regions.star,
                            &c.maximum(&regions.right_fan, &regions.right_exterior)?,
                        )?,
                    )?,
                )?;
                let complete = c.mul(&covered, &c.mul(&sides[0].valid, &sides[1].valid)?)?;
                let zero = Tensor::zeros(complete.shape(), a.dtype(), c.device())?;
                let status_value = |value: i64| {
                    Tensor::full_scalar(complete.shape(), DType::I64, value as f64, c.device())
                };
                let unavailable = c.eq(
                    &star.status,
                    &status_value(WetStarStatus::Unavailable as i64)?,
                )?;
                let status = c.where_cond(
                    &unavailable,
                    &status_value(WetWaveStatus::Unavailable as i64)?,
                    &c.where_cond(
                        &complete,
                        &status_value(WetWaveStatus::Certified as i64)?,
                        &status_value(WetWaveStatus::Unresolved as i64)?,
                    )?,
                )?;
                for mask in [
                    &mut regions.left_exterior,
                    &mut regions.left_fan,
                    &mut regions.star,
                    &mut regions.right_fan,
                    &mut regions.right_exterior,
                ] {
                    *mask = c.where_cond(&complete, mask, &zero)?;
                }
                Ok(WetWaveDomain {
                    star,
                    endpoint_sound,
                    sides,
                    regions,
                    status,
                })
            })?;
            result.push(domain);
        }
        Ok(result)
    }
}

#[cfg(test)]
#[path = "wet_waves_tests.rs"]
mod tests;
