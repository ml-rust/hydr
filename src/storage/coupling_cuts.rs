//! Model-bound attachment cuts and original-water donor availability.

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

use super::coupling_cut_bounds::CutArithmetic;
use super::{LevelPoolBatch, LevelPoolError};
use crate::storage::tensor::TensorPoolContext;

#[derive(Clone, Copy, Debug)]
pub(crate) struct LevelPoolCut {
    pub pool_index: usize,
    pub elevation_m: f64,
}

#[derive(Debug)]
pub(crate) struct LevelPoolCutPlan<R: Runtime<DType = DType>> {
    context: Arc<TensorPoolContext<R>>,
    indices: Tensor<R>,
    #[cfg(test)]
    nominal_m3: Tensor<R>,
    #[cfg(test)]
    lower_m3: Tensor<R>,
    upper_m3: Tensor<R>,
}

pub(crate) struct LevelPoolCutAvailability<R: Runtime<DType = DType>> {
    pub available_lower_m3: Tensor<R>,
}

impl<R: Runtime<DType = DType>> LevelPoolCutPlan<R> {
    #[cfg(test)]
    pub(crate) fn nominal_volume_m3(&self) -> &Tensor<R> {
        &self.nominal_m3
    }

    #[cfg(test)]
    pub(crate) fn lower_volume_m3(&self) -> &Tensor<R> {
        &self.lower_m3
    }

    pub(crate) fn upper_volume_m3(&self) -> &Tensor<R> {
        &self.upper_m3
    }
}

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
    pub(crate) fn bind_coupling_cuts(
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

    pub(crate) fn coupling_cut_availability(
        &self,
        plan: &LevelPoolCutPlan<R>,
    ) -> Result<LevelPoolCutAvailability<R>, LevelPoolError> {
        if !Arc::ptr_eq(&self.context, &plan.context) {
            return Err(LevelPoolError::input(
                "cut plan belongs to another model/client. Bind cuts on this pool batch",
            ));
        }
        if plan.upper_m3.numel() == 0 {
            return Ok(LevelPoolCutAvailability {
                available_lower_m3: plan.upper_m3.clone(),
            });
        }
        let c = &self.context.client;
        let original = c.index_select(&self.state.volume_m3, 0, &plan.indices)?;
        let arithmetic = CutArithmetic {
            context: &self.context,
        };
        let lower = arithmetic.add(&original, &c.neg(&plan.upper_m3)?, false)?;
        let zero = c.mul_scalar(&lower, 0.0)?;
        Ok(LevelPoolCutAvailability {
            available_lower_m3: c.maximum(&lower, &zero)?,
        })
    }
}

#[cfg(test)]
#[path = "coupling_cut_tests.rs"]
mod tests;
