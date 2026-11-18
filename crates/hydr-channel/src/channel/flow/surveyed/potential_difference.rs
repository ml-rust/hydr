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

#[cfg(test)]
mod tests {
    use super::super::error::{PrismaticRiemannError, into_public};
    use super::super::integration::potential;
    use super::super::potential_difference_test_data::{CASES, GRAVITY, PROFILES};
    use super::super::potential_difference_test_support::{batch, contains, target, values};
    use super::*;
    use numr::ops::{HypotOps, MidpointOps};

    fn fallback<R>(
        geometry: &HydraulicGeometry<R>,
        bands: &GatheredBands<R>,
        original: &Tensor<R>,
        target: &Interval<R>,
        config: &PrismaticRiemannConfig,
        work: &WorkLedger,
    ) -> HydraulicResult<Interval<R>>
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
        let mut endpoints = Vec::new();
        for area in [original, target.lower(), target.upper()] {
            let state = checked(c, "fallback endpoint recovery", |a| {
                // Public entry points capture contiguous conserved operands before private recovery.
                let captured = a.point(area)?;
                geometry.recover(a, bands, captured.lower(), &config.recovery)
            })?;
            endpoints.push(potential(geometry, bands, &state, config, 4096, work)?);
        }
        checked(c, "complete endpoint potential subtraction", |a| {
            let full = a.bounds(endpoints[1].lower(), endpoints[2].upper())?;
            Ok(a.sub(&full, &endpoints[0])?)
        })
    }

    fn references<R>(client: R::Client) -> HydraulicResult<()>
    where
        R: Runtime<DType = DType>,
        R::Client: TensorOps<R>
            + HypotOps<R>
            + MidpointOps<R>
            + CompareOps<R>
            + NextRepresentableOps<R>
            + ScalarOps<R>
            + TwoSumOps<R>
            + RuntimeClient<R>
            + CertifiedSimpsonAlgorithms<R>,
    {
        assert_eq!(CASES.len(), 18);
        let config = PrismaticRiemannConfig {
            gravity_m_per_s2: f64::from_bits(GRAVITY),
            ..Default::default()
        };
        for strided in [false, true] {
            for case in CASES {
                let batch = batch::<R>(&client, &PROFILES[case.profile])?;
                let geometry = HydraulicGeometry::from_batch(&batch);
                let owners = values::<R, _>(&client, &[0_i64; 3], strided)?;
                let bands = checked(&client, "repeated original section owners", |a| {
                    geometry.gather(a, &owners)
                })?;
                let original =
                    values::<R, _>(&client, &[f64::from_bits(case.original); 3], strided)?;
                let target = target::<R>(&client, case.target, strided)?;
                let work = WorkLedger::new(&config);
                let direct =
                    direct_increment(&geometry, &bands, &original, &target, &config, 4096, &work)?;
                let available = direct.available.try_to_vec::<f64>()?;
                let required = case.gate != "containment_only";
                if required {
                    assert_eq!(
                        available, [1.0; 3],
                        "{} direct admission, strided={strided}",
                        case.id
                    );
                }
                if matches!(
                    case.id,
                    "knot_overlap"
                        | "knot_crossing"
                        | "dry_target"
                        | "dry_overlap"
                        | "wide_positive_span"
                ) {
                    assert_eq!(available, [0.0; 3], "{} fallback admission", case.id);
                    assert_eq!(
                        work.snapshot().simpson_value_points,
                        0,
                        "{} inactive callbacks",
                        case.id
                    );
                    assert_eq!(
                        work.snapshot().simpson_derivative_points,
                        0,
                        "{} inactive derivatives",
                        case.id
                    );
                }
                let result = if available.iter().all(|&value| value == 1.0) {
                    direct.bounds
                } else {
                    fallback(&geometry, &bands, &original, &target, &config, &work)?
                };
                contains(
                    &result,
                    case.reference,
                    case.id,
                    case.gate == "common_band_width",
                )?;
                for endpoint in case.endpoints {
                    contains(&result, endpoint, case.id, false)?;
                }
                if case.gate == "exact_zero" {
                    assert_eq!(result.lower().try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(result.upper().try_to_vec::<f64>()?, [0.0; 3]);
                    assert_eq!(work.snapshot().simpson_value_points, 0);
                    assert_eq!(work.snapshot().simpson_derivative_points, 0);
                }
                if case.id == "straddling_original" {
                    assert!(
                        result
                            .lower()
                            .try_to_vec::<f64>()?
                            .iter()
                            .all(|&value| value < 0.0)
                    );
                    assert!(
                        result
                            .upper()
                            .try_to_vec::<f64>()?
                            .iter()
                            .all(|&value| value > 0.0)
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn signed_potential_increments_enclose_independent_references_cpu()
    -> Result<(), PrismaticRiemannError> {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), None));
        references::<CpuRuntime>(client).map_err(|error| into_public(error, Default::default()))
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn signed_potential_increments_enclose_independent_references_cuda()
    -> Result<(), Box<dyn std::error::Error>> {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        references::<CudaRuntime>(CudaClient::new(CudaDevice::new(0))?)
            .map_err(|error| into_public(error, Default::default()))?;
        Ok(())
    }
}
