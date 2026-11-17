//! Per-row source geometry of one fixed paired source, kept over every target callback.
//!
//! A signed potential increment needs the source top width and the analytic source terms.
//! They depend only on the fixed source state and its bands, never on the target. Each row
//! evaluates once, through the same calls on a row subset. Every call acts row by row, so a
//! kept row matches a fresh evaluation in every bit.

use super::analytic_bottom::AnalyticSource;
use super::potential_absolute::admitted_analytic;
use super::potential_range::{PotentialContext, select_expansion, source_query};
use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult, checked};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, MidpointOps, NextRepresentableOps, ScalarOps, SortingOps,
        TensorOps, TwoSumOps, UtilityOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};
use std::cell::RefCell;

/// Source terms of the rows of one paired side, in parent row order.
pub(super) struct SourceTerms<R: Runtime<DType = DType>> {
    /// Parent row count.
    count: usize,
    kept: RefCell<Option<Kept<R>>>,
}

/// Dense parent-row storage. `ready` is `1` where a row holds its evaluated terms.
struct Kept<R: Runtime<DType = DType>> {
    ready: Tensor<R>,
    width: [Tensor<R>; 2],
    masks: [Tensor<R>; 3],
    expansions: [[Tensor<R>; 3]; 3],
}

/// Source top width and analytic source terms of selected rows.
pub(super) struct SelectedTerms<R: Runtime<DType = DType>> {
    pub(super) top_width: Interval<R>,
    pub(super) analytic: AnalyticSource<R>,
}

/// A context's rows as parent rows of one paired side.
pub(super) struct SourceRows<'a, R: Runtime<DType = DType>> {
    pub(super) terms: &'a SourceTerms<R>,
    /// Parent row of each context row.
    pub(super) parents: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SourceTerms<R> {
    pub(super) fn new(count: usize) -> Self {
        Self {
            count,
            kept: RefCell::new(None),
        }
    }
}

impl<R> SourceRows<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    /// The same parent side at the context rows `rows`.
    pub(super) fn select(&self, client: &R::Client, rows: &Tensor<R>) -> HydraulicResult<Self> {
        Ok(Self {
            terms: self.terms,
            parents: client.index_select(&self.parents, 0, rows)?,
        })
    }
}

impl<R> SourceRows<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + MidpointOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Evaluates every missing row of the context rows `rows` of `context`.
    pub(super) fn ensure(
        &self,
        context: &PotentialContext<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<()> {
        let parent_count = self.terms.count;
        let c = context.geometry.client();
        let mut kept = self.terms.kept.try_borrow_mut().map_err(|_| overlap())?;
        // A failed evaluation keeps the earlier rows and stores nothing new.
        let parents = c.index_select(&self.parents, 0, rows)?;
        let missing = match kept.as_ref() {
            Some(kept) => {
                let zero = Tensor::zeros(&[], DType::F64, c.device())?;
                let ready = c.index_select(&kept.ready, 0, &parents)?;
                let local = c.nonzero(&c.eq(&ready, &zero)?)?.flatten()?;
                c.index_select(rows, 0, &local)?
            }
            None => rows.clone(),
        };
        if missing.numel() == 0 {
            return Ok(());
        }
        let (bands, source, original_rows) = checked(c, "source terms selection", |a| {
            Ok((
                context.bands.select(a, &missing)?,
                context.source.select(a, &missing)?,
                c.index_select(&context.query.query.original_rows, 0, &missing)?,
            ))
        })?;
        let query = source_query(context.geometry, &bands, &source, &original_rows)?;
        let gravity = context.evaluation.config.gravity_m_per_s2;
        let parents = c.index_select(&self.parents, 0, &missing)?;
        let filled = checked(c, "source terms evaluation", |a| {
            // A fresh increment selects its rows from the whole source query, as here.
            let all = c.arange(0.0, missing.numel() as f64, 1.0, DType::I64)?;
            let query = query.select(a, &all)?;
            let area = select_expansion(a, context.source.requested_areas_m2(), &missing)?;
            let state = context.geometry.recover_query(a, &bands, &query)?.state;
            let analytic = admitted_analytic(a, &bands, &area, &state.depth, gravity)?;
            let dense = match kept.as_ref() {
                Some(dense) => dense.put(a, &parents, &state.top_width, &analytic)?,
                None => Kept::zeros(a, parent_count, &state.top_width, &analytic)?.put(
                    a,
                    &parents,
                    &state.top_width,
                    &analytic,
                )?,
            };
            Ok(dense)
        })?;
        *kept = Some(filled);
        Ok(())
    }

    /// Kept terms of the context rows `rows`. Call [`Self::ensure`] on them first.
    pub(super) fn terms(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: &Tensor<R>,
    ) -> HydraulicResult<SelectedTerms<R>> {
        let c = a.client();
        let kept = self.terms.kept.try_borrow().map_err(|_| overlap())?;
        let kept = kept
            .as_ref()
            .ok_or_else(|| invalid("terms requested before their rows were evaluated"))?;
        let parents = c.index_select(&self.parents, 0, rows)?;
        let zero = Tensor::zeros(&[], DType::F64, c.device())?;
        let ready = c.index_select(&kept.ready, 0, &parents)?;
        if c.nonzero(&c.eq(&ready, &zero)?)?.numel() != 0 {
            return Err(invalid("terms requested before their rows were evaluated"));
        }
        let take = |value: &Tensor<R>| c.index_select(value, 0, &parents);
        let expansion = |a: &mut Arithmetic<'_, R>, parts: &[Tensor<R>; 3]| {
            HydraulicResult::Ok(a.expansion_bounds(
                &take(&parts[0])?,
                &take(&parts[1])?,
                &take(&parts[2])?,
            )?)
        };
        let [eligible, rectangular, pointed] = &kept.masks;
        let [potential, sound, sound_squared] = &kept.expansions;
        Ok(SelectedTerms {
            top_width: a.bounds(&take(&kept.width[0])?, &take(&kept.width[1])?)?,
            analytic: AnalyticSource {
                eligible: take(eligible)?,
                rectangular: take(rectangular)?,
                pointed: take(pointed)?,
                potential: expansion(a, potential)?,
                sound: expansion(a, sound)?,
                sound_squared: expansion(a, sound_squared)?,
            },
        })
    }
}

impl<R> Kept<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Zero storage for `count` rows in the element types of `width` and `analytic`.
    fn zeros(
        a: &Arithmetic<'_, R>,
        count: usize,
        width: &Interval<R>,
        analytic: &AnalyticSource<R>,
    ) -> HydraulicResult<Self> {
        let device = a.client().device();
        let zero = |like: &Tensor<R>| Tensor::<R>::zeros(&[count], like.dtype(), device);
        let parts = |value: &CertifiedExpansion<R>| {
            HydraulicResult::Ok([
                zero(value.high())?,
                zero(value.low())?,
                zero(value.abs_error())?,
            ])
        };
        Ok(Self {
            ready: Tensor::zeros(&[count], DType::F64, device)?,
            width: [zero(width.lower())?, zero(width.upper())?],
            masks: [
                zero(&analytic.eligible)?,
                zero(&analytic.rectangular)?,
                zero(&analytic.pointed)?,
            ],
            expansions: [
                parts(&analytic.potential)?,
                parts(&analytic.sound)?,
                parts(&analytic.sound_squared)?,
            ],
        })
    }

    /// Storage with the evaluated terms of `parents` written and marked ready.
    fn put(
        &self,
        a: &Arithmetic<'_, R>,
        parents: &Tensor<R>,
        width: &Interval<R>,
        analytic: &AnalyticSource<R>,
    ) -> HydraulicResult<Self> {
        let c = a.client();
        let put = |dense: &Tensor<R>, value: &Tensor<R>| c.index_put(dense, 0, parents, value);
        let parts = |dense: &[Tensor<R>; 3], value: &CertifiedExpansion<R>| {
            HydraulicResult::Ok([
                put(&dense[0], value.high())?,
                put(&dense[1], value.low())?,
                put(&dense[2], value.abs_error())?,
            ])
        };
        let ones = Tensor::ones(parents.shape(), DType::F64, c.device())?;
        let [eligible, rectangular, pointed] = &self.masks;
        let [potential, sound, sound_squared] = &self.expansions;
        Ok(Self {
            ready: put(&self.ready, &ones)?,
            width: [
                put(&self.width[0], width.lower())?,
                put(&self.width[1], width.upper())?,
            ],
            masks: [
                put(eligible, &analytic.eligible)?,
                put(rectangular, &analytic.rectangular)?,
                put(pointed, &analytic.pointed)?,
            ],
            expansions: [
                parts(potential, &analytic.potential)?,
                parts(sound, &analytic.sound)?,
                parts(sound_squared, &analytic.sound_squared)?,
            ],
        })
    }
}

fn overlap() -> HydraulicError {
    invalid("borrow overlaps another evaluation")
}

fn invalid(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Source terms {message}. Discard the paired source graph"),
    }
}
