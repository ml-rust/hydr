use super::{
    paired_source_cache::{self, SourcePotentialCache},
    paired_sources::PairedSources,
    potential_range::PotentialEvaluation,
    source_binding::SourcePanelLevel,
    wave_curve_types::{CurveValue, unavailable},
};
use crate::channel::flow::surveyed::error::{HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{CompareOps, MidpointOps, NextRepresentableOps, ScalarOps, TensorOps, TwoSumOps},
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use row_admission::{admit_rows, storage, storage_error};
use solvr::integrate::CertifiedSimpsonAlgorithms;
use std::sync::Arc;
use values::owned_values;

#[path = "paired_source_panel_admission.rs"]
mod row_admission;
#[path = "paired_source_panel_values.rs"]
mod values;

/// Owned checked potentials and their authenticated source/panel policy.
pub(super) struct SourcePanelValues<R: Runtime<DType = DType>> {
    pub(super) values: [CurveValue<R>; 2],
    pub(super) level: SourcePanelLevel,
    pub(super) parent_rows: [Tensor<R>; 2],
    side: [usize; 2],
}

impl<R> PairedSources<'_, R>
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
    pub(super) fn panel_level(&self, index: usize) -> HydraulicResult<SourcePanelLevel> {
        self.root_witness()?.panel_level(index)
    }

    pub(super) fn source_values_at_level(
        &mut self,
        level: &SourcePanelLevel,
        rows: [&Tensor<R>; 2],
    ) -> HydraulicResult<SourcePanelValues<R>> {
        self.check_root_witness(level.witness())?;
        admit_rows(self, rows)?;
        let c = self.geometry.client();
        if rows.iter().all(|rows| rows.numel() == 0) {
            let values = checked(c, "empty source panel request", |a| {
                Ok([unavailable(a, 0)?, unavailable(a, 0)?])
            })?;
            return owned_values(c, level, rows, values);
        }
        if level.index() == 0 {
            let values = self.source_values(rows)?;
            return owned_values(c, level, rows, values);
        }
        let position = self
            .panel_cache
            .iter()
            .position(|(index, _)| *index == level.index());
        let position = if let Some(position) = position {
            position
        } else {
            let count = self.original_rows.numel();
            checked::<R, _>(c, "source panel cache storage admission", |a| {
                let mut levels = 1usize;
                let mut panels = self.evaluation.panels;
                for _ in 0..self.evaluation.config.max_refinements {
                    let Some(next) = panels.checked_mul(2) else {
                        break;
                    };
                    if next > self.evaluation.config.max_simpson_panels {
                        break;
                    }
                    panels = next;
                    levels = levels
                        .checked_add(1)
                        .ok_or_else(|| storage_error(count, a.dtype(), levels))?;
                }
                storage::<R>(count, a.dtype(), levels)?;
                Ok(())
            })?;
            let caches = checked::<R, _>(c, "source panel cache allocation", |a| {
                Ok([
                    SourcePotentialCache::with_query(a, Arc::clone(&self.cache[0].query), count)?,
                    SourcePotentialCache::with_query(a, Arc::clone(&self.cache[1].query), count)?,
                ])
            })?;
            self.panel_cache.try_reserve_exact(1).map_err(|_| {
                storage_error(
                    count,
                    self.cache[0].ready.dtype(),
                    self.panel_cache.len() + 1,
                )
            })?;
            self.panel_cache.push((level.index(), caches));
            self.panel_cache.len() - 1
        };
        let evaluation = PotentialEvaluation {
            config: self.evaluation.config,
            panels: level.panels(),
            work: self.evaluation.work,
        };
        let [left, right] = &mut self.panel_cache[position].1;
        let values = [
            paired_source_cache::ensure(
                left,
                self.geometry,
                self.bands,
                self.sources[0],
                self.boundaries,
                &evaluation,
                rows[0],
            )?,
            paired_source_cache::ensure(
                right,
                self.geometry,
                self.bands,
                self.sources[1],
                self.boundaries,
                &evaluation,
                rows[1],
            )?,
        ];
        owned_values(c, level, rows, values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::flow::surveyed::error::HydraulicError;
    use crate::channel::flow::surveyed_points::{
        chart_fixture,
        source_fixture_support::{Client, Result, components, endpoints, hydraulic},
    };
    use solvr::optimize::CertifiedBisectionAlgorithms;

    mod admission {
        use super::*;

        pub(super) fn run<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let config = chart_fixture::config();
            let id = ["scalar_trapezoid_depth_jump"];
            let foreign = chart_fixture::with_pair::<R, _>(client, &id, &config, |pair, _| {
                hydraulic(pair.panel_level(1))
            })?;
            chart_fixture::with_pair::<R, _>(client, &id, &config, |pair, work| {
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                let level = hydraulic(pair.panel_level(1))?;
                let before = work.snapshot();
                let coarse = chart_fixture::cache(pair)?;
                assert!(
                    pair.source_values_at_level(&foreign, [&rows, &rows])
                        .is_err()
                );
                let rejected_level = config.max_refinements + 1;
                let HydraulicError::Numerical { context: error } = pair
                    .panel_level(rejected_level)
                    .err()
                    .ok_or("Out-of-limit panel level accepted")?
                else {
                    return Err("Out-of-limit panel level returned a non-numerical error".into());
                };
                assert!(error.contains(&format!("level={rejected_level}")));
                assert!(error.contains(&format!("max_refinements={}", config.max_refinements)));
                assert!(error.contains(&format!("max_panels={}", config.max_simpson_panels)));
                assert!(pair.panel_level(usize::MAX).is_err());
                let panels = pair.evaluation.panels;
                for changed in [0, panels * 2] {
                    pair.evaluation.panels = changed;
                    assert!(pair.source_values_at_level(&level, [&rows, &rows]).is_err());
                    assert!(pair.panel_level(0).is_err());
                }
                pair.evaluation.panels = panels;
                hydraulic(pair.source_values_at_level(&level, [&empty, &empty]))?;
                for (position, invalid) in [
                    Tensor::from_slice(&[-1_i64], &[1], client.device())?,
                    Tensor::from_slice(&[1_i64], &[1], client.device())?,
                    Tensor::from_slice(&[i64::MAX], &[1], client.device())?,
                    rows.reshape(&[1, 1])?,
                    Tensor::from_slice(&[0_f64], &[1], client.device())?,
                ]
                .into_iter()
                .enumerate()
                {
                    let HydraulicError::Numerical { context: error } = pair
                        .source_values_at_level(&level, [&rows, &invalid])
                        .err()
                        .ok_or("Invalid right source panel request accepted")?
                    else {
                        return Err(
                            "Invalid right source panel request returned a non-numerical error"
                                .into(),
                        );
                    };
                    assert!(error.contains("side=right"));
                    if position < 3 {
                        let parent = [-1_i64, 1, i64::MAX][position];
                        assert!(error.contains(&format!("parent={parent}")));
                        assert!(error.contains("captured_count=1"));
                        assert!(error.contains("[0, 1)"));
                    } else {
                        assert!(error.contains("rank="));
                        assert!(error.contains("dtype="));
                        assert!(error.contains("device="));
                    }
                    let base = hydraulic(pair.panel_level(0))?;
                    assert!(
                        pair.source_values_at_level(&base, [&rows, &invalid])
                            .is_err()
                    );
                }
                assert!(pair.panel_cache.is_empty());
                assert_eq!(work.snapshot(), before);
                assert_eq!(chart_fixture::cache(pair)?, coarse);
                Ok(())
            })?;
            let huge = crate::channel::flow::surveyed::PrismaticRiemannConfig {
                max_refinements: usize::MAX,
                max_simpson_panels: 1usize << (usize::BITS - 1),
                ..config
            };
            chart_fixture::with_pair::<R, _>(client, &id, &huge, |pair, work| {
                let before = work.snapshot();
                assert!(pair.panel_level(usize::MAX).is_err());
                assert!(pair.panel_level(usize::BITS as usize).is_err());
                assert!(pair.panel_cache.is_empty());
                assert_eq!(work.snapshot(), before);
                Ok(())
            })?;
            Ok(())
        }

        #[cfg(feature = "cuda")]
        pub(super) fn foreign_device(client: &numr::runtime::cuda::CudaClient) -> Result {
            use numr::runtime::cuda::{CudaDevice, CudaRuntime};
            let foreign = Tensor::<CudaRuntime>::empty(&[0], DType::I64, &CudaDevice::new(1))?;
            chart_fixture::with_pair::<CudaRuntime, _>(
                client,
                &["scalar_trapezoid_depth_jump"],
                &chart_fixture::config(),
                |pair, work| {
                    let before = work.snapshot();
                    let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                    let level = hydraulic(pair.panel_level(1))?;
                    assert!(
                        pair.source_values_at_level(&level, [&empty, &foreign])
                            .is_err()
                    );
                    assert!(pair.panel_cache.is_empty());
                    assert_eq!(work.snapshot(), before);
                    Ok(())
                },
            )
        }
    }

    mod behavior {
        use super::*;
        use crate::channel::flow::surveyed_points::source_potential_fixtures as fixtures;

        pub(super) fn truth_and_dedup<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let ids = [
                "scalar_trapezoid_depth_jump",
                "scalar_three_bands_depth_jump",
            ];
            let config = chart_fixture::config();
            chart_fixture::with_pair::<R, _>(client, &ids, &config, |pair, work| {
                let rows = Tensor::from_slice(&[1_i64, 0, 1], &[3], client.device())?;
                let unique = Tensor::from_slice(&[0_i64, 1], &[2], client.device())?;
                let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                let coarse = hydraulic(pair.panel_level(0))?;
                let fine = hydraulic(pair.panel_level(4))?;
                let low = hydraulic(pair.source_values_at_level(&coarse, [&rows, &rows]))?;
                let before = work.snapshot();
                let high = hydraulic(pair.source_values_at_level(&fine, [&rows, &rows]))?;
                let after = work.snapshot();
                assert!(after.simpson_value_points > before.simpson_value_points);
                assert!(after.simpson_derivative_points > before.simpson_derivative_points);
                assert_eq!(after.residual_points, before.residual_points);
                for side in 0..2 {
                    let lo = endpoints(&low.values[side].physical)?;
                    let hi = endpoints(&high.values[side].physical)?;
                    let retained = endpoints(
                        &crate::channel::flow::surveyed_points::source_fixture_support::bounds::<R>(
                            client,
                            &high.values[side].retained,
                        )?,
                    )?;
                    for (position, parent) in [1usize, 0, 1].into_iter().enumerate() {
                        let truth = fixtures::CASES
                            .iter()
                            .find(|case| case.id == ids[parent])
                            .ok_or("Missing frozen panel source truth")?;
                        assert_eq!(truth.gravity_bits, config.gravity_m_per_s2.to_bits());
                        let expected = truth.sides[side].phi.bounds.map(f64::from_bits);
                        for actual in [lo[position], hi[position], retained[position]] {
                            assert!(
                                actual[0] <= expected[0] && expected[1] <= actual[1],
                                "{} side={side}: {actual:?} excludes {expected:?}",
                                ids[parent]
                            );
                        }
                        assert!(
                            hi[position][1] - hi[position][0] < lo[position][1] - lo[position][0]
                        );
                    }
                    assert_eq!(hi[0], hi[2]);
                }
                let high_before = components(&high.values[0].retained)?;
                let baseline = chart_fixture::cache(pair)?;
                let cached = hydraulic(pair.source_values_at_level(&fine, [&unique, &unique]))?;
                assert_eq!(work.snapshot(), after);
                assert_eq!(components(&high.values[0].retained)?, high_before);
                assert_eq!(
                    endpoints(&cached.values[0].physical)?[1],
                    endpoints(&high.values[0].physical)?[0]
                );
                assert_eq!(chart_fixture::cache(pair)?, baseline);
                let caches = pair.panel_cache.len();
                let unused = hydraulic(pair.panel_level(5))?;
                hydraulic(pair.source_values_at_level(&unused, [&empty, &empty]))?;
                assert_eq!(pair.panel_cache.len(), caches);
                assert_eq!(work.snapshot(), after);
                Ok(())
            })?;
            let mut totals = Vec::new();
            for values in [&[1_i64, 0, 1][..], &[0_i64, 1][..]] {
                totals.push(chart_fixture::with_pair::<R, _>(
                    client,
                    &ids,
                    &config,
                    |pair, work| {
                        let rows = Tensor::from_slice(values, &[values.len()], client.device())?;
                        let level = hydraulic(pair.panel_level(1))?;
                        hydraulic(pair.source_values_at_level(&level, [&rows, &rows]))?;
                        Ok(work.snapshot())
                    },
                )?);
            }
            assert_eq!(totals[0], totals[1]);
            Ok(())
        }

        pub(super) fn work_error<R>(client: &R::Client) -> Result
        where
            R: Runtime<DType = DType>,
            R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
        {
            let mut config = chart_fixture::config();
            let id = ["scalar_trapezoid_depth_jump"];
            let charge = chart_fixture::with_pair::<R, _>(client, &id, &config, |pair, work| {
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                let level = hydraulic(pair.panel_level(1))?;
                hydraulic(pair.source_values_at_level(&level, [&rows, &empty]))?;
                let work = work.snapshot();
                Ok(work.simpson_value_points + work.simpson_derivative_points)
            })?;
            assert!(charge > 0);
            config.max_native_query_points = charge;
            chart_fixture::with_pair::<R, _>(client, &id, &config, |pair, work| {
                let rows = Tensor::from_slice(&[0_i64], &[1], client.device())?;
                let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                let level = hydraulic(pair.panel_level(1))?;
                let result = pair.source_values_at_level(&level, [&rows, &rows]);
                assert!(matches!(
                    result,
                    Err(HydraulicError::Rejected {
                        reason:
                            crate::channel::flow::surveyed::PrismaticRiemannRejection::WorkLimit,
                        ..
                    })
                ));
                assert_eq!(pair.panel_cache.len(), 1);
                let caches = &pair.panel_cache[0].1;
                assert_eq!(caches[0].ready.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(caches[1].ready.try_to_vec::<f64>()?, [0.0]);
                let before = work.snapshot();
                assert_eq!(
                    before.simpson_value_points + before.simpson_derivative_points,
                    charge
                );
                let checked = hydraulic(pair.source_values_at_level(&level, [&rows, &empty]))?;
                assert_eq!(checked.values[0].available.try_to_vec::<f64>()?, [1.0]);
                assert_eq!(work.snapshot(), before);
                assert!(pair.source_values_at_level(&level, [&rows, &rows]).is_err());
                assert_eq!(work.snapshot(), before);
                Ok(())
            })
        }
    }

    fn basic<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
    {
        let config = chart_fixture::config();
        chart_fixture::with_pair::<R, _>(
            client,
            &["scalar_rectangle_depth_jump"],
            &config,
            |pair, work| {
                let rows = Tensor::from_slice(&[0_i64, 0], &[2], client.device())?;
                let empty = Tensor::empty(&[0], DType::I64, client.device())?;
                let level0 = hydraulic(pair.panel_level(0))?;
                let level1 = hydraulic(pair.panel_level(1))?;
                assert_eq!(level0.index(), 0);
                assert_eq!(level1.panels(), 2 * level0.panels());
                let original = hydraulic(pair.source_values([&rows, &empty]))?;
                let baseline = chart_fixture::cache(pair)?;
                let cached = hydraulic(pair.source_values_at_level(&level0, [&rows, &empty]))?;
                assert_eq!(
                    components(&original[0].retained)?,
                    components(&cached.values[0].retained)?
                );
                assert_eq!(
                    endpoints(&original[0].physical)?,
                    endpoints(&cached.values[0].physical)?
                );
                let higher = hydraulic(pair.source_values_at_level(&level1, [&rows, &empty]))?;
                assert_eq!(higher.level.index(), 1);
                assert_eq!(pair.panel_cache.len(), 1);
                assert_eq!(chart_fixture::cache(pair)?, baseline);
                let before = work.snapshot();
                let repeated = hydraulic(pair.source_values_at_level(&level1, [&rows, &empty]))?;
                assert_eq!(
                    components(&higher.values[0].retained)?,
                    components(&repeated.values[0].retained)?
                );
                assert_eq!(work.snapshot(), before);
                assert_eq!(higher.values[0].available.try_to_vec::<f64>()?, [1.0, 1.0]);
                Ok(())
            },
        )
    }

    fn matrix<R>(client: &R::Client) -> Result
    where
        R: Runtime<DType = DType>,
        R::Client: Client<R> + CertifiedSimpsonAlgorithms<R> + CertifiedBisectionAlgorithms<R>,
    {
        basic::<R>(client)?;
        behavior::truth_and_dedup::<R>(client)?;
        behavior::work_error::<R>(client)?;
        admission::run::<R>(client)?;
        assert!(storage::<R>(usize::MAX, DType::F64, 2).is_err());
        assert!(storage::<R>(1, DType::F64, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn configured_cpu_source_panel_levels_preserve_coarse_cache() -> Result {
        use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(2), Some(257)));
        matrix::<CpuRuntime>(&client)
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn actual_cuda_source_panel_levels_preserve_coarse_cache() -> Result {
        use numr::runtime::cuda::{CudaClient, CudaDevice, CudaRuntime};
        let client = CudaClient::new(CudaDevice::new(0))?;
        matrix::<CudaRuntime>(&client)?;
        admission::foreign_device(&client)
    }
}
