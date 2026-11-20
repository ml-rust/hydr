//! Per-row target geometry of one paired endpoint, kept over both source sides.
//!
//! Both sides of a paired endpoint read the geometry of the same singleton target query.
//! Each row recovers once, through the same call on a row subset. Recovery acts row by row,
//! so a kept row matches a fresh recovery in every bit.

use crate::channel::flow::surveyed::error::{Arithmetic, HydraulicError, HydraulicResult};
use crate::channel::section::{Constitutive, GatheredBands, HydraulicGeometry, SupportedAreaQuery};
use numr::{
    dtype::DType,
    ops::{
        CompareOps, IndexingOps, NextRepresentableOps, ScalarOps, SortingOps, TensorOps, TwoSumOps,
    },
    runtime::{Runtime, RuntimeClient},
    tensor::Tensor,
};
use solvr::common::arithmetic::Interval;
use std::cell::RefCell;

/// Target geometry of the rows of one singleton target query, in its row order.
pub(super) struct TargetStates<'q, R: Runtime<DType = DType>> {
    geometry: &'q HydraulicGeometry<R>,
    bands: &'q GatheredBands<R>,
    query: &'q SupportedAreaQuery<R>,
    kept: RefCell<Option<Kept<R>>>,
}

/// Dense row storage. `ready` is `1` where a row holds its recovered geometry.
struct Kept<R: Runtime<DType = DType>> {
    ready: Tensor<R>,
    /// Depth, forward area, area, top width, integral and hydraulic depth bounds.
    intervals: [[Tensor<R>; 2]; 6],
    /// Wet and dry possibility masks.
    masks: [Tensor<R>; 2],
}

/// A context's rows as rows of one kept target.
pub(super) struct TargetRows<'a, R: Runtime<DType = DType>> {
    pub(super) states: &'a TargetStates<'a, R>,
    /// Target row of each context row.
    pub(super) rows: Tensor<R>,
}

impl<'q, R: Runtime<DType = DType>> TargetStates<'q, R> {
    /// `query` is a singleton endpoint query over `bands`. Each context bound to it reads a
    /// row selection of this query and of these bands.
    pub(super) fn new(
        geometry: &'q HydraulicGeometry<R>,
        bands: &'q GatheredBands<R>,
        query: &'q SupportedAreaQuery<R>,
    ) -> Self {
        Self {
            geometry,
            bands,
            query,
            kept: RefCell::new(None),
        }
    }
}

impl<R> TargetRows<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: IndexingOps<R>,
{
    /// The same target at the context rows `rows`.
    pub(super) fn select(&self, client: &R::Client, rows: &Tensor<R>) -> HydraulicResult<Self> {
        Ok(Self {
            states: self.states,
            rows: client.index_select(&self.rows, 0, rows)?,
        })
    }
}

impl<R> TargetRows<'_, R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R>
        + CompareOps<R>
        + NextRepresentableOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + RuntimeClient<R>,
{
    /// Target geometry of the context rows `rows`, or of every context row when `None`.
    /// Rows without kept geometry recover first.
    pub(super) fn state(
        &self,
        a: &mut Arithmetic<'_, R>,
        rows: Option<&Tensor<R>>,
    ) -> HydraulicResult<Constitutive<R>> {
        let states = self.states;
        let c = a.client();
        let parents = match rows {
            Some(rows) => c.index_select(&self.rows, 0, rows)?,
            None => self.rows.clone(),
        };
        let mut kept = states.kept.try_borrow_mut().map_err(|_| overlap())?;
        let missing = match kept.as_ref() {
            Some(kept) => {
                let zero = Tensor::zeros(&[], DType::F64, c.device())?;
                let ready = c.index_select(&kept.ready, 0, &parents)?;
                let local = c.nonzero(&c.eq(&ready, &zero)?)?.flatten()?;
                c.index_select(&parents, 0, &local)?
            }
            None => parents.clone(),
        };
        if missing.numel() != 0 {
            let bands = states.bands.select(a, &missing)?;
            let query = states.query.select(a, &missing)?;
            let state = states.geometry.recover_query(a, &bands, &query)?.state;
            let dense = match kept.take() {
                Some(dense) => dense,
                None => Kept::zeros(states.bands.section_owner.numel(), &state, c)?,
            };
            *kept = Some(dense.put(c, &missing, &state)?);
        }
        let kept = kept
            .as_ref()
            .ok_or_else(|| invalid("geometry requested before its rows were recovered"))?;
        let take = |value: &Tensor<R>| c.index_select(value, 0, &parents);
        let mut intervals = Vec::with_capacity(kept.intervals.len());
        for [lower, upper] in &kept.intervals {
            intervals.push(a.bounds(&take(lower)?, &take(upper)?)?);
        }
        let [
            depth,
            forward_area,
            area,
            top_width,
            integral,
            hydraulic_depth,
        ]: [Interval<R>; 6] = intervals
            .try_into()
            .map_err(|_| invalid("kept geometry lost an interval"))?;
        Ok(Constitutive {
            depth,
            forward_area,
            area,
            top_width,
            integral,
            hydraulic_depth,
            wet_possible: take(&kept.masks[0])?,
            dry_possible: take(&kept.masks[1])?,
        })
    }
}

impl<R> Kept<R>
where
    R: Runtime<DType = DType>,
    R::Client: TensorOps<R> + RuntimeClient<R>,
{
    /// Zero storage for `count` rows in the element types of `state`.
    fn zeros(count: usize, state: &Constitutive<R>, c: &R::Client) -> HydraulicResult<Self> {
        let device = c.device();
        let zero = |like: &Tensor<R>| Tensor::<R>::zeros(&[count], like.dtype(), device);
        let pair =
            |value: &Interval<R>| HydraulicResult::Ok([zero(value.lower())?, zero(value.upper())?]);
        Ok(Self {
            ready: Tensor::zeros(&[count], DType::F64, device)?,
            intervals: [
                pair(&state.depth)?,
                pair(&state.forward_area)?,
                pair(&state.area)?,
                pair(&state.top_width)?,
                pair(&state.integral)?,
                pair(&state.hydraulic_depth)?,
            ],
            masks: [zero(&state.wet_possible)?, zero(&state.dry_possible)?],
        })
    }

    /// Storage with the recovered geometry of `rows` written and marked ready.
    fn put(self, c: &R::Client, rows: &Tensor<R>, state: &Constitutive<R>) -> HydraulicResult<Self>
    where
        R::Client: IndexingOps<R>,
    {
        let put = |dense: &Tensor<R>, value: &Tensor<R>| c.index_put(dense, 0, rows, value);
        let pair = |dense: &[Tensor<R>; 2], value: &Interval<R>| {
            HydraulicResult::Ok([
                put(&dense[0], value.lower())?,
                put(&dense[1], value.upper())?,
            ])
        };
        let ones = Tensor::ones(rows.shape(), DType::F64, c.device())?;
        let [
            depth,
            forward_area,
            area,
            top_width,
            integral,
            hydraulic_depth,
        ] = &self.intervals;
        Ok(Self {
            ready: put(&self.ready, &ones)?,
            intervals: [
                pair(depth, &state.depth)?,
                pair(forward_area, &state.forward_area)?,
                pair(area, &state.area)?,
                pair(top_width, &state.top_width)?,
                pair(integral, &state.integral)?,
                pair(hydraulic_depth, &state.hydraulic_depth)?,
            ],
            masks: [
                put(&self.masks[0], &state.wet_possible)?,
                put(&self.masks[1], &state.dry_possible)?,
            ],
        })
    }
}

fn overlap() -> HydraulicError {
    invalid("borrow overlaps another evaluation")
}

fn invalid(message: &str) -> HydraulicError {
    HydraulicError::Numerical {
        context: format!("Target states {message}. Discard the paired endpoint"),
    }
}
