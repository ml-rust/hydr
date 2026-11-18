use std::collections::BTreeMap;
use std::sync::Arc;

use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, IndexingOps, ReduceOps, ScalarOps, ShapeOps, SortingOps,
    TwoSumOps, TypeConversionOps, UnaryOps,
};
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::interpolate::{Interp1dAlgorithms, InterpMethod};

use super::{LevelPoolCut, LevelPoolCutPlan};
use crate::storage::coupling_cut_bounds::CutArithmetic;
use crate::storage::{LevelPoolBatch, LevelPoolError};

struct BoundCut {
    position: usize,
    elevation: f64,
    knot_or_segment: usize,
    exact: bool,
}

impl<R: Runtime<DType = DType>> LevelPoolBatch<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + IndexingOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + ShapeOps<R>
        + SortingOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>
        + Interp1dAlgorithms<R>,
{
    #[doc(hidden)]
    pub fn bind_coupling_cuts(
        &self,
        cuts: &[LevelPoolCut],
    ) -> Result<LevelPoolCutPlan<R>, LevelPoolError> {
        let context = &self.context;
        let c = &context.client;
        let keys = self
            .elements()
            .iter()
            .map(|e| e.curve.key())
            .collect::<std::collections::BTreeSet<_>>();
        let groups = keys
            .into_iter()
            .enumerate()
            .map(|(index, key)| (key, index))
            .collect::<BTreeMap<_, _>>();
        let mut grouped = (0..context.groups.len())
            .map(|_| Vec::new())
            .collect::<Vec<Vec<BoundCut>>>();
        let mut pool_indices = Vec::with_capacity(cuts.len());
        for (position, cut) in cuts.iter().enumerate() {
            let element = self.elements().get(cut.pool_index).ok_or_else(|| {
                LevelPoolError::input(format!(
                    "cut pool_index={}: supply an existing pool index",
                    cut.pool_index
                ))
            })?;
            let knots = element.curve.knots();
            if !cut.elevation_m.is_finite()
                || cut.elevation_m < knots[0].stage_m
                || cut.elevation_m > knots[knots.len() - 1].stage_m
            {
                return Err(LevelPoolError::input(format!(
                    "cut elevation={}: supply a finite elevation within the closed curve support",
                    cut.elevation_m
                ))
                .at(cut.pool_index));
            }
            let right = knots.partition_point(|k| k.stage_m < cut.elevation_m);
            let exact = knots[right].stage_m == cut.elevation_m;
            let knot_or_segment = if exact { right } else { right - 1 };
            let group = groups.get(&element.curve.key()).ok_or_else(|| {
                LevelPoolError::numerical("cut curve lacks its cached native group")
            })?;
            grouped[*group].push(BoundCut {
                position,
                elevation: cut.elevation_m,
                knot_or_segment,
                exact,
            });
            pool_indices.push(
                i64::try_from(cut.pool_index)
                    .map_err(|_| LevelPoolError::input("cut pool index exceeds I64 indexing"))?,
            );
        }
        let indices = Tensor::<R>::from_slice(&pool_indices, &[cuts.len()], &context.device)?;
        if cuts.is_empty() {
            let empty = context.tensor(&[])?;
            return Ok(LevelPoolCutPlan {
                context: Arc::clone(context),
                indices,
                #[cfg(test)]
                nominal_m3: empty.clone(),
                #[cfg(test)]
                lower_m3: empty.clone(),
                upper_m3: empty,
            });
        }
        #[cfg(test)]
        let mut nominal = Vec::new();
        #[cfg(test)]
        let mut lower = Vec::new();
        let mut upper = Vec::new();
        let mut inverse = vec![0i64; cuts.len()];
        let mut next = 0usize;
        let arithmetic = CutArithmetic { context };
        for (group, records) in context.groups.iter().zip(grouped) {
            for exact in [true, false] {
                let records = records
                    .iter()
                    .filter(|r| r.exact == exact)
                    .collect::<Vec<_>>();
                if records.is_empty() {
                    continue;
                }
                let positions = records
                    .iter()
                    .map(|r| {
                        i64::try_from(r.knot_or_segment).map_err(|_| {
                            LevelPoolError::input("cut knot index exceeds I64 indexing")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let knot_indices =
                    Tensor::<R>::from_slice(&positions, &[positions.len()], &context.device)?;
                let v0 = c.index_select(&group.volumes, 0, &knot_indices)?;
                if exact {
                    #[cfg(test)]
                    nominal.push(v0.clone());
                    #[cfg(test)]
                    lower.push(v0.clone());
                    upper.push(v0);
                } else {
                    let positions = positions.iter().map(|index| index + 1).collect::<Vec<_>>();
                    let right_indices =
                        Tensor::<R>::from_slice(&positions, &[positions.len()], &context.device)?;
                    let v1 = c.index_select(&group.volumes, 0, &right_indices)?;
                    let h0 = c.index_select(&group.stages, 0, &knot_indices)?;
                    let h1 = c.index_select(&group.stages, 0, &right_indices)?;
                    let query =
                        context.tensor(&records.iter().map(|r| r.elevation).collect::<Vec<_>>())?;
                    let value =
                        c.interp1d(&group.stages, &group.volumes, &query, InterpMethod::Linear)?;
                    context.finite(&value, "nominal cut inverse volume")?;
                    let (lo, hi) = arithmetic.inverse(&query, &h0, &h1, &v0, &v1)?;
                    context.reject(&c.gt(&lo, &hi)?, "cut inverse enclosure is unresolved")?;
                    #[cfg(test)]
                    nominal.push(value);
                    #[cfg(test)]
                    lower.push(lo);
                    upper.push(hi);
                }
                for record in records {
                    inverse[record.position] = i64::try_from(next)
                        .map_err(|_| LevelPoolError::input("cut count exceeds I64 indexing"))?;
                    next += 1;
                }
            }
        }
        let inverse = Tensor::<R>::from_slice(&inverse, &[inverse.len()], &context.device)?;
        let ordered = |values: Vec<Tensor<R>>| -> Result<Tensor<R>, LevelPoolError> {
            Ok(c.index_select(&c.cat(&values.iter().collect::<Vec<_>>(), 0)?, 0, &inverse)?)
        };
        Ok(LevelPoolCutPlan {
            context: Arc::clone(context),
            indices,
            #[cfg(test)]
            nominal_m3: ordered(nominal)?,
            #[cfg(test)]
            lower_m3: ordered(lower)?,
            upper_m3: ordered(upper)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::coupling_cuts::fixture::batch;
    use crate::storage::{LevelPoolConfig, LevelPoolState};
    use numr::runtime::cpu::{CpuClient, CpuDevice, CpuRuntime, ParallelismConfig};

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn cuts_retain_knots_and_enclose_affine_inverse_in_requested_order() -> TestResult {
        let batch = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 8.0)?;
        let plan = batch.bind_coupling_cuts(&[
            LevelPoolCut {
                pool_index: 0,
                elevation_m: 1.5,
            },
            LevelPoolCut {
                pool_index: 0,
                elevation_m: 0.0,
            },
            LevelPoolCut {
                pool_index: 0,
                elevation_m: -2.0,
            },
            LevelPoolCut {
                pool_index: 0,
                elevation_m: 4.0,
            },
        ])?;
        let nominal = plan.nominal_volume_m3().try_to_vec::<f64>()?;
        let lower = plan.lower_volume_m3().try_to_vec::<f64>()?;
        let upper = plan.upper_volume_m3().try_to_vec::<f64>()?;
        assert_eq!(nominal, vec![6.0, 3.0, 0.0, 11.0]);
        assert!(lower[0] <= 6.0 && upper[0] >= 6.0);
        for index in 1..4 {
            assert_eq!(lower[index].to_bits(), nominal[index].to_bits());
            assert_eq!(upper[index].to_bits(), nominal[index].to_bits());
        }
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?;
        assert!(available[0] > 0.0 && available[0] <= 2.0);
        assert_eq!(&available[1..], &[5.0, 8.0, 0.0]);
        Ok(())
    }

    #[test]
    fn cuts_reject_foreign_models_and_unsupported_inputs() -> TestResult {
        let original = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
        let foreign = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
        let plan = original.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: 1.0,
        }])?;
        assert!(matches!(
            foreign.coupling_cut_availability(&plan),
            Err(LevelPoolError::InvalidInput { .. })
        ));
        for elevation_m in [-1.0, 3.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                original.bind_coupling_cuts(&[LevelPoolCut {
                    pool_index: 0,
                    elevation_m
                }]),
                Err(LevelPoolError::InvalidInput { .. })
            ));
        }
        assert!(matches!(
            original.bind_coupling_cuts(&[LevelPoolCut {
                pool_index: 1,
                elevation_m: 1.0
            }]),
            Err(LevelPoolError::InvalidInput { .. })
        ));
        assert!(original.clone().coupling_cut_availability(&plan).is_ok());
        Ok(())
    }

    #[test]
    fn empty_cuts_preserve_empty_native_support() -> TestResult {
        let batch = batch(&[0.0, 4.0], &[0.0, 2.0], 2.0)?;
        let plan = batch.bind_coupling_cuts(&[])?;
        assert_eq!(plan.upper_volume_m3().shape(), &[0]);
        assert_eq!(
            batch
                .coupling_cut_availability(&plan)?
                .available_lower_m3
                .shape(),
            &[0]
        );
        Ok(())
    }

    #[test]
    fn interleaved_cuts_preserve_pool_order_across_repeated_curve_groups() -> TestResult {
        let first = batch(&[0.0, 3.0, 11.0], &[-2.0, 0.0, 4.0], 8.0)?;
        let second = batch(&[0.0, 8.0], &[0.0, 4.0], 6.0)?;
        let mut elements = vec![
            first.elements()[0].clone(),
            second.elements()[0].clone(),
            first.elements()[0].clone(),
        ];
        for (index, element) in elements.iter_mut().enumerate() {
            element.id = format!("pool-{index}");
            element.storage_support_identity = format!("finite-storage-{index}");
        }
        let client = CpuClient::new(CpuDevice::new())
            .with_parallelism(ParallelismConfig::new(Some(1), Some(64)));
        let batch = LevelPoolBatch::<CpuRuntime>::new(
            client,
            elements,
            vec![
                LevelPoolState { volume_m3: 8.0 },
                LevelPoolState { volume_m3: 6.0 },
                LevelPoolState { volume_m3: 10.0 },
            ],
            0.0,
            LevelPoolConfig::default(),
        )?;
        let cuts = [
            LevelPoolCut {
                pool_index: 1,
                elevation_m: 1.5,
            },
            LevelPoolCut {
                pool_index: 2,
                elevation_m: 0.0,
            },
            LevelPoolCut {
                pool_index: 0,
                elevation_m: 1.5,
            },
            LevelPoolCut {
                pool_index: 1,
                elevation_m: 4.0,
            },
            LevelPoolCut {
                pool_index: 2,
                elevation_m: 2.5,
            },
            LevelPoolCut {
                pool_index: 0,
                elevation_m: -2.0,
            },
        ];
        let plan = batch.bind_coupling_cuts(&cuts)?;
        let expected = [3.0, 3.0, 6.0, 8.0, 8.0, 0.0];
        let lower = plan.lower_volume_m3().try_to_vec::<f64>()?;
        let upper = plan.upper_volume_m3().try_to_vec::<f64>()?;
        assert_eq!(plan.nominal_volume_m3().try_to_vec::<f64>()?, expected);
        for ((lower, upper), expected) in lower.iter().zip(&upper).zip(expected) {
            assert!(*lower <= expected && *upper >= expected);
        }
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?;
        for (value, ceiling) in available.iter().zip([3.0, 7.0, 2.0, 0.0, 2.0, 8.0]) {
            assert!(*value >= 0.0 && *value <= ceiling);
            if ceiling > 0.0 {
                assert!(*value > 0.0);
            }
        }
        assert_eq!(available[1], 7.0);
        assert_eq!(available[3], 0.0);
        assert_eq!(available[5], 8.0);
        Ok(())
    }

    #[test]
    fn nonrepresentable_affine_inverse_contains_adjacent_binary64_bounds() -> TestResult {
        let batch = batch(&[0.0, 3.5], &[0.125, 0.875], 3.5)?;
        let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: 0.375,
        }])?;
        // Exact V = 7/6 lies strictly between these adjacent binary64 operands.
        let exact_lower = f64::from_bits(0x3ff2_aaaa_aaaa_aaaa);
        let exact_upper = f64::from_bits(0x3ff2_aaaa_aaaa_aaab);
        let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
        let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
        assert!(lower <= exact_lower);
        assert!(upper >= exact_upper);
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?[0];
        // Exact eligibility = 7/3 exceeds this operand and is below its successor.
        let eligibility_lower = f64::from_bits(0x4002_aaaa_aaaa_aaaa);
        assert!(available > 0.0 && available <= eligibility_lower);
        Ok(())
    }

    #[test]
    fn subnormal_equal_scale_cut_encloses_exact_unit_slope_volume() -> TestResult {
        let top = 1.0e-308;
        let cut = 5.0e-309;
        let batch = batch(&[0.0, top], &[0.0, top], top)?;
        let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: cut,
        }])?;
        let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
        let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
        assert!(lower.is_finite() && lower >= 0.0 && lower <= cut);
        assert!(upper.is_finite() && upper >= cut && upper <= top);
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?[0];
        assert!(available > 0.0 && available <= top - cut);
        Ok(())
    }

    #[test]
    fn near_overflow_equal_scale_cut_encloses_exact_unit_slope_volume() -> TestResult {
        let top = 1.0e308;
        let cut = 5.0e307;
        let batch = batch(&[0.0, top], &[0.0, top], top)?;
        let plan = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: cut,
        }])?;
        let lower = plan.lower_volume_m3().try_to_vec::<f64>()?[0];
        let upper = plan.upper_volume_m3().try_to_vec::<f64>()?[0];
        assert!(lower.is_finite() && lower >= 0.0 && lower <= cut);
        assert!(upper.is_finite() && upper >= cut && upper <= top);
        let available = batch
            .coupling_cut_availability(&plan)?
            .available_lower_m3
            .try_to_vec::<f64>()?[0];
        assert!(available > 0.0 && available <= cut);
        Ok(())
    }

    #[test]
    fn nonfinite_directed_cut_intermediate_returns_typed_numerical_error() -> TestResult {
        let batch = batch(&[0.0, f64::MAX], &[0.0, f64::MAX], 0.0)?;
        let cut = f64::from_bits(f64::MAX.to_bits() - 1);
        let result = batch.bind_coupling_cuts(&[LevelPoolCut {
            pool_index: 0,
            elevation_m: cut,
        }]);
        assert!(matches!(result, Err(LevelPoolError::Numerical { .. })));
        Ok(())
    }
}
