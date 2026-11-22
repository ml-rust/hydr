//! Runoff into surface cells: the exchange lanes, the per-lane runoff sums, and the surface
//! completion with runoff.
//!
//! The exchange lanes hold the link cells in link-lane order, then every runoff-only cell in
//! ascending order. Donor groups keep the link lanes only, so incoming runoff funds no
//! withdrawal. Runoff joins the exchange inflow after transport with zero velocity. A lane with
//! link inflow keeps the link momentum, so its velocity is that momentum over the link and
//! runoff inflow together.
//!
//! The delivery lanes of one exchange lane add in delivery-lane order. One delivery lane keeps
//! its volume in every bit. The sum of `k` parts lies within `γ_{k-1} Σ parts` of the exact
//! sum, with `γ_n = nε / (1 − nε)`.

use super::admission::RunoffTarget;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::solver::Solver;
use hydr_solver::internal::{
    CouplingCompletion, ExchangeSource, SurfaceCouplingStage, SurfaceLinks,
};
use numr::dtype::DType;
use numr::ops::{BinaryOps, CompareOps, ConditionalOps, IndexingOps, ShapeOps};
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;
use std::collections::BTreeSet;

/// Inflow, withdrawal, and x/y inflow momentum of every link lane, each `[S]`.
pub(in super::super) type LinkLanes<R> = [Tensor<R>; 4];

/// Static surface delivery operands.
pub(in super::super) struct SurfaceRunoff<R: Runtime<DType = DType>> {
    /// Exchange lanes: the `S` link cells, then the runoff-only cells.
    pub links: SurfaceLinks<R>,
    /// Exchange lane of every delivery lane, or `None` off the surface.
    pub lane_of: Vec<Option<usize>>,
    /// Delivery lanes of every exchange lane, in delivery-lane order.
    pub members: Vec<Vec<usize>>,
    /// Gathers of the delivery volumes, one per addend rank, I64 `[S + R]` each.
    ///
    /// Gather `k` picks the `k`-th delivery lane of each exchange lane, or the appended zero.
    pub gathers: Vec<Tensor<R>>,
    /// Delivery lanes into the surface, ascending, I64 `[Ns]`.
    pub delivery: Tensor<R>,
}

impl<R: Runtime<DType = DType>> SurfaceRunoff<R> {
    /// Surface operands of the delivery `targets` over the link cells `link_cells` of
    /// `surface`, or `None` without a surface target.
    pub fn new(
        client: &R::Client,
        surface: &Solver<R>,
        link_cells: &[usize],
        targets: &[RunoffTarget],
    ) -> Result<Option<Self>, SurfaceChannelError> {
        let mut extra = BTreeSet::new();
        for target in targets {
            if let RunoffTarget::Surface { cell } = *target
                && link_cells.binary_search(&cell).is_err()
            {
                extra.insert(cell);
            }
        }
        let receiving: Vec<usize> = targets
            .iter()
            .enumerate()
            .filter_map(|(lane, target)| {
                matches!(target, RunoffTarget::Surface { .. }).then_some(lane)
            })
            .collect();
        if receiving.is_empty() {
            return Ok(None);
        }
        let extra: Vec<usize> = extra.into_iter().collect();
        let link_lanes = link_cells.len();
        let lane_of: Vec<Option<usize>> = targets
            .iter()
            .map(|target| match *target {
                RunoffTarget::Surface { cell } => link_cells
                    .binary_search(&cell)
                    .ok()
                    .or_else(|| extra.binary_search(&cell).ok().map(|k| link_lanes + k)),
                _ => None,
            })
            .collect();
        let cells: Vec<usize> = link_cells.iter().chain(&extra).copied().collect();
        let mut members = vec![Vec::new(); cells.len()];
        for (lane, exchange) in lane_of.iter().enumerate() {
            if let Some(slot) = exchange.and_then(|exchange| members.get_mut(exchange)) {
                slot.push(lane);
            }
        }
        let indices = |values: &[usize]| -> Result<Tensor<R>, SurfaceChannelError> {
            let values = values
                .iter()
                .map(|&v| {
                    i64::try_from(v).map_err(|_| {
                        SurfaceChannelError::input("surface runoff lanes exceed I64 indexing")
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            Tensor::<R>::from_slice(&values, &[values.len()], client.device()).map_err(Into::into)
        };
        let ranks = members.iter().map(Vec::len).max().unwrap_or(0);
        let zero = targets.len();
        let gathers = (0..ranks)
            .map(|rank| {
                let picks: Vec<usize> = members
                    .iter()
                    .map(|lanes| lanes.get(rank).copied().unwrap_or(zero))
                    .collect();
                indices(&picks)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Some(Self {
            links: surface.coupling_links(&cells)?,
            lane_of,
            members,
            gathers,
            delivery: indices(&receiving)?,
        }))
    }

    /// Exchange lane count `S + R`.
    pub fn lane_count(&self) -> usize {
        self.links.len()
    }

    /// Upper bound on the summation error of every exchange lane, in m³.
    ///
    /// `parts` holds the delivered volume of every delivery lane.
    pub fn sum_bound(&self, parts: &[f64]) -> f64 {
        let mut total = 0.0_f64;
        for lanes in &self.members {
            if lanes.len() < 2 {
                continue;
            }
            let width = (lanes.len() - 1) as f64 * f64::EPSILON;
            let magnitude: f64 = lanes
                .iter()
                .map(|&lane| parts.get(lane).copied().unwrap_or(f64::NAN).abs())
                .sum();
            let bound = (width * magnitude / (1.0 - width)).next_up();
            total = (total + bound).next_up();
        }
        total
    }
}

impl<R: Runtime<DType = DType>> SurfaceRunoff<R>
where
    R::Client: BinaryOps<R> + IndexingOps<R> + ShapeOps<R>,
{
    /// Runoff inflow of every exchange lane from the `delivered` volumes `[N]`, `[S + R]`.
    pub fn lane_inflow(
        &self,
        client: &R::Client,
        delivered: &Tensor<R>,
    ) -> Result<Tensor<R>, SurfaceChannelError> {
        let zero = Tensor::<R>::zeros(&[1], DType::F64, client.device())?;
        let padded = client.cat(&[delivered, &zero], 0)?;
        let mut sum: Option<Tensor<R>> = None;
        for gather in &self.gathers {
            let part = client.index_select(&padded, 0, gather)?;
            sum = Some(match sum {
                Some(sum) => client.add(&sum, &part)?,
                None => part,
            });
        }
        sum.ok_or_else(|| {
            SurfaceChannelError::Numerical("surface runoff lanes hold no delivery lane".into())
        })
    }
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Surface delivery operands of this coordinator, or `None` without a surface target.
    pub fn surface_runoff(&self) -> Option<&SurfaceRunoff<R>> {
        self.runoff.as_ref()?.surface.as_ref()
    }

    /// Exchange lanes of the surface: the link lanes, then the runoff-only lanes.
    ///
    /// Returns `None` without link lanes and without surface runoff.
    pub fn exchange_links(&self) -> Option<&SurfaceLinks<R>> {
        self.surface_runoff()
            .map(|surface| &surface.links)
            .or(self.links.as_ref())
    }

    /// The link lanes `[S]` of an exchange-lane operand `values`.
    pub fn link_lanes(&self, values: &Tensor<R>) -> Result<Tensor<R>, SurfaceChannelError> {
        let s = self.surface_count;
        if values.shape() == [s] {
            return Ok(values.clone());
        }
        if s == 0 {
            return Ok(Tensor::<R>::zeros(&[0], DType::F64, self.client.device())?);
        }
        Ok(values.narrow(0, 0, s)?)
    }

    /// Completes `surface` over `stage` with the link lane operands `link` and the runoff inflow
    /// `runoff` of every exchange lane.
    ///
    /// `link` is `None` without link lanes. Runoff-only lanes take zero link water and zero
    /// momentum. Without exchange lanes the surface completes alone.
    pub fn complete_exchange(
        &self,
        surface: &Solver<R>,
        stage: &SurfaceCouplingStage<R>,
        link: Option<LinkLanes<R>>,
        runoff: Option<&Tensor<R>>,
    ) -> Result<Solver<R>, SurfaceChannelError> {
        let Some(links) = self.exchange_links() else {
            return self.complete_alone(surface, stage);
        };
        let c = &self.client;
        let lanes = links.len();
        let extra = lanes.saturating_sub(self.surface_count);
        let zeros = |count: usize| Tensor::<R>::zeros(&[count], DType::F64, c.device());
        let pad = |values: Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            if extra == 0 {
                return Ok(values);
            }
            Ok(c.cat(&[&values, &zeros(extra)?], 0)?)
        };
        let [inflow, withdrawal, momentum_x, momentum_y] = match link {
            Some([inflow, withdrawal, x, y]) => [pad(inflow)?, pad(withdrawal)?, pad(x)?, pad(y)?],
            None => {
                let zero = zeros(lanes)?;
                [zero.clone(), zero.clone(), zero.clone(), zero]
            }
        };
        let inflow = match runoff {
            Some(runoff) => c.add(&inflow, runoff)?,
            None => inflow,
        };
        let zero = self.scalar(0.0)?;
        let has = c.gt(&inflow, &zero)?;
        let safe = c.where_cond(&has, &inflow, &self.scalar(1.0)?)?;
        let velocity = |momentum: &Tensor<R>| -> Result<Tensor<R>, SurfaceChannelError> {
            Ok(c.where_cond(&has, &c.div(momentum, &safe)?, &zero)?)
        };
        let source = ExchangeSource::new(
            links,
            inflow.contiguous()?,
            withdrawal.contiguous()?,
            [velocity(&momentum_x)?, velocity(&momentum_y)?],
        )?;
        match surface.coupling_complete(stage, Some(&source))? {
            CouplingCompletion::Accepted(next) => Ok(*next),
            CouplingCompletion::Retry(reason) => Err(SurfaceChannelError::Numerical(format!(
                "surface completion: {reason}. Shorten the common interval"
            ))),
        }
    }
}
