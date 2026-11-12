use super::PrismaticRiemannConfig;
use super::equation_ops::choose;
use super::error::{HydraulicResult, checked};
use super::integration::{complete_integration, retain_cause};
use super::potential_difference_derivatives::{integrand, panel_fourth_bound};
use super::potential_difference_lanes::DifferenceLanes;
use super::work::{QueryWork, WorkLedger};
use crate::channel::section::{GatheredBands, HydraulicGeometry};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use solvr::common::arithmetic::Interval;
use solvr::integrate::{
    CertifiedSimpsonAlgorithms, CertifiedSimpsonOptions, CertifiedSimpsonValues,
};
use std::cell::Cell;

pub(crate) struct DirectIncrement<R: Runtime<DType = DType>> {
    pub bounds: Interval<R>,
    pub available: Tensor<R>,
}

pub(crate) fn direct_increment<R>(
    geometry: &HydraulicGeometry<R>,
    bands: &GatheredBands<R>,
    original_area: &Tensor<R>,
    target_area: &Interval<R>,
    config: &PrismaticRiemannConfig,
    panels: usize,
    work: &WorkLedger,
) -> HydraulicResult<DirectIncrement<R>>
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
    let c = geometry.client();
    let lanes = checked(c, "surveyed potential difference lanes", |a| {
        DifferenceLanes::build(
            a,
            bands,
            original_area,
            target_area,
            config.gravity_m_per_s2,
        )
    })?;
    integrate_lanes(geometry, &lanes, config, panels, work)
}

pub(crate) fn integrate_lanes<R>(
    geometry: &HydraulicGeometry<R>,
    lanes: &DifferenceLanes<R>,
    config: &PrismaticRiemannConfig,
    panels: usize,
    work: &WorkLedger,
) -> HydraulicResult<DirectIncrement<R>>
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
    let c = geometry.client();
    let left = Tensor::zeros(&[lanes.active.numel()], DType::F64, c.device())?;
    let cause = Cell::new(None);
    let result = c.certified_simpson(
        &left,
        &lanes.active,
        CertifiedSimpsonOptions {
            panels,
            max_points: config.max_native_query_points,
        },
        |query| {
            let result = work
                .charge(QueryWork::SimpsonValue, query.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed potential difference values", |a| {
                        let s = a.bounds(&query.lower, &query.upper)?;
                        let value = integrand(a, lanes, &query.owner, &s)?;
                        let point = a.point(&query.point)?;
                        let nominal = integrand(a, lanes, &query.owner, &point)?;
                        let half = c.mul_scalar(&c.sub(nominal.upper(), nominal.lower())?, 0.5)?;
                        let nominal = c.add(nominal.lower(), &half)?;
                        let nominal =
                            c.maximum(value.lower(), &c.minimum(&nominal, value.upper())?)?;
                        Ok(CertifiedSimpsonValues {
                            value: nominal,
                            lower: value.lower().clone(),
                            upper: value.upper().clone(),
                        })
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
        |panel| {
            let result = work
                .charge(QueryWork::SimpsonDerivative, panel.owner.numel())
                .and_then(|()| {
                    checked(c, "surveyed potential difference fourth derivatives", |a| {
                        let s = a.bounds(&panel.lower, &panel.upper)?;
                        panel_fourth_bound(a, lanes, &panel.owner, &s)
                    })
                });
            result.map_err(|error| retain_cause(&cause, error))
        },
    );
    let result = complete_integration(result, cause.take())?;
    checked(c, "surveyed potential difference assembly", |a| {
        let shape = [lanes.face_count, lanes.band_count];
        let lower = result.lower().reshape(&shape)?;
        let upper = result.upper().reshape(&shape)?;
        let active = lanes.active.reshape(&shape)?;
        let empty = Tensor::zeros(&[lanes.face_count], DType::F64, c.device())?;
        let mut total = a.point(&empty)?;
        let mut available = lanes.equal.clone();
        for band in 0..lanes.band_count {
            let mask = active.narrow(1, band, 1)?.squeeze(Some(1));
            let term = a.bounds(
                &lower.narrow(1, band, 1)?.squeeze(Some(1)),
                &upper.narrow(1, band, 1)?.squeeze(Some(1)),
            )?;
            // Nondegenerate certified band interiors are disjoint, so each face selects at most one term.
            total = choose(a, &mask, &term, &total)?;
            available = c.maximum(&available, &mask)?;
        }
        Ok(DirectIncrement {
            bounds: total,
            available,
        })
    })
}
