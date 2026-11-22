//! Runoff pairing and composite balance terms.
//!
//! Per interval, each receiving binding must hold its delivered volume in every bit. Over the
//! lifetime, the runoff child's outlet export, the coordinator lane counters, and the receiving
//! ledgers must agree within their bounds.
//!
//! In the composite balance the runoff child joins with its storage, rain, losses, and soil
//! adjustment. Delivered runoff is an internal transfer: it leaves the receivers' external
//! inflow totals.

use super::attempt::RunoffAttempt;
use super::counters::RunoffCounters;
use crate::channel::ChannelReach;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::coupling::surface_channels::pools::{PoolChild, inflow_error};
use crate::runoff::SubcatchmentBatch;
use numr::dtype::DType;
use numr::ops::{ReduceOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Summation allowance added to every lane count, in units of `f64::EPSILON`.
const SUM_SLACK: f64 = 8.0;

/// Runoff terms of the composite balance, in m³.
#[derive(Clone, Copy, Debug, Default)]
pub(in super::super) struct RunoffTerms {
    pub initial: f64,
    pub storage: f64,
    /// Applied rain on the runoff support.
    pub rain: f64,
    /// Infiltration plus evaporation.
    pub losses: f64,
    /// Signed soil water adjustment.
    pub adjustment: f64,
    /// Lifetime runoff the receivers recorded.
    pub delivered: f64,
    /// Runoff residual and rounding bounds, pairing bounds included.
    pub bound: f64,
    pub magnitude: f64,
}

/// Every runoff child and counter state the balance reads.
pub(in super::super) type RunoffState<'a, R> =
    Option<(&'a SubcatchmentBatch<R>, &'a RunoffCounters<R>)>;

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Sum of `values`, zero when empty.
    fn sum(&self, values: &Tensor<R>) -> Result<f64, SurfaceChannelError> {
        if values.numel() == 0 {
            return Ok(0.0);
        }
        Ok(self
            .client
            .sum(&values.contiguous()?, &[0], false)?
            .item::<f64>()?)
    }

    /// Runoff each reach with runoff bindings recorded on them, with the reach index.
    fn reach_runoff(
        &self,
        reaches: &[ChannelReach<R>],
    ) -> Result<Vec<(usize, ReachRunoff<R>)>, SurfaceChannelError> {
        let topology = self.runoff_topology()?;
        let mut parts = Vec::new();
        for (reach, own) in topology.reaches.iter().enumerate() {
            if own.is_empty() {
                continue;
            }
            let child = reaches.get(reach).ok_or_else(|| {
                SurfaceChannelError::Numerical("a runoff reach is missing".into())
            })?;
            let accounting = child.coupling_accounting();
            let offset = topology.offsets.get(reach).copied().unwrap_or_default();
            let count = own.len();
            parts.push((
                reach,
                ReachRunoff {
                    lifetime: accounting.lateral_volume_m3.narrow(0, offset, count)?,
                    bound: accounting
                        .lateral_counter_roundoff_bound_m3
                        .narrow(0, offset, count)?,
                    interval: accounting
                        .lateral_interval_volume_m3
                        .map(|values| values.narrow(0, offset, count))
                        .transpose()?,
                },
            ));
        }
        Ok(parts)
    }

    /// Rejects an interval whose receiving bindings differ from the delivered volumes in any bit.
    ///
    /// `reaches` holds the candidate reaches in setup order. `pool_received` holds the original
    /// interval volume of every runoff pool binding, or `None` without pool lanes.
    pub fn check_runoff_delivery(
        &self,
        attempt: &RunoffAttempt<R>,
        reaches: &[ChannelReach<R>],
        pool_received: Option<&Tensor<R>>,
    ) -> Result<(), SurfaceChannelError> {
        let topology = self.runoff_topology()?;
        let mut tensors = Vec::new();
        let mut lanes = Vec::new();
        for (reach, part) in self.reach_runoff(reaches)? {
            let interval = part.interval.ok_or_else(|| {
                SurfaceChannelError::Numerical("a runoff reach lacks its lateral receipt".into())
            })?;
            tensors.push(interval);
            lanes.extend(topology.reaches[reach].iter().map(|binding| binding.lane));
        }
        match (pool_received, topology.pool_lanes.is_empty()) {
            (Some(received), false) => {
                tensors.push(received.clone());
                lanes.extend(topology.pool_lanes.iter().copied());
            }
            (None, true) => {}
            _ => {
                return Err(SurfaceChannelError::Numerical(
                    "runoff pool receipts differ from the runoff pool lanes".into(),
                ));
            }
        }
        if tensors.is_empty() {
            return Ok(());
        }
        let received = self
            .client
            .cat(&tensors.iter().collect::<Vec<_>>(), 0)?
            .try_to_vec::<f64>()?;
        if received.len() != lanes.len() {
            return Err(SurfaceChannelError::Numerical(
                "runoff receipts differ from the delivery lanes".into(),
            ));
        }
        for (&value, &lane) in received.iter().zip(&lanes) {
            let expected = attempt.delivered_m3.get(lane).copied().unwrap_or(f64::NAN);
            if value.to_bits() != expected.to_bits() {
                let node = topology
                    .nodes
                    .get(lane)
                    .map(|node| node.identity())
                    .unwrap_or_default();
                return Err(SurfaceChannelError::Numerical(format!(
                    "runoff pairing: {node} received={value}, delivered={expected}: a receiver accepted a runoff volume that differs from the coordinator volume"
                )));
            }
        }
        Ok(())
    }

    /// Runoff terms of the children `reaches` and `pools`, or `None` without runoff.
    ///
    /// Rejects a nonzero runon ledger, and lifetime exports, lane counters, and receiving
    /// ledgers that disagree beyond their bounds.
    pub fn runoff_terms(
        &self,
        runoff: RunoffState<'_, R>,
        reaches: &[ChannelReach<R>],
        pools: Option<&PoolChild<R>>,
    ) -> Result<Option<RunoffTerms>, SurfaceChannelError> {
        let Some((child, counters)) = runoff else {
            return Ok(None);
        };
        let ledger = child.ledger();
        if ledger.external_runon_volume_m3 != 0.0 {
            return Err(SurfaceChannelError::input(format!(
                "runoff-external-runon: runoff child runon={}: the coordinator admits no external runon",
                ledger.external_runon_volume_m3
            )));
        }
        let bounds = child
            .accounting_bounds()
            .map_err(SurfaceChannelError::Runoff)?;
        let (mut delivered, mut delivered_bound) = (0.0, 0.0);
        for (_, part) in self.reach_runoff(reaches)? {
            delivered += self.sum(&part.lifetime)?;
            delivered_bound += self.sum(&part.bound)?;
        }
        if let (Some(pools), Some(topology)) = (pools, &self.pools) {
            let native = topology
                .inflows
                .native_summary(&self.client, &pools.external)
                .map_err(inflow_error)?;
            delivered += native.volume_m3;
            delivered_bound += native.roundoff_bound_m3 + native.reduction_bound_m3;
        }
        let counted = self.sum(&counters.delivered_m3)?;
        let counted_bound = self.sum(&counters.roundoff_bound_m3)?;
        let lanes = self.runoff_topology()?.lane_count() as f64;
        let width = (lanes + SUM_SLACK) * f64::EPSILON;
        let outlet = ledger.outlet_volume_m3;
        let export_bound = bounds.outlet_error_m3
            + counted_bound
            + counters.split_bound_m3
            + width * (outlet + counted);
        if (outlet - counted).abs() > export_bound {
            return Err(SurfaceChannelError::Numerical(format!(
                "runoff lifetime pairing: outlet export={outlet}, lane counters={counted}, bound={export_bound}: the runoff child and the delivery counters disagree"
            )));
        }
        let receipt_bound = delivered_bound + counted_bound + width * (delivered + counted);
        if (delivered - counted).abs() > receipt_bound {
            return Err(SurfaceChannelError::Numerical(format!(
                "runoff lifetime pairing: receivers={delivered}, lane counters={counted}, bound={receipt_bound}: the receivers and the delivery counters disagree"
            )));
        }
        let losses = ledger.infiltration_volume_m3 + ledger.evaporation_volume_m3;
        Ok(Some(RunoffTerms {
            initial: ledger.initial_storage_m3,
            storage: ledger.storage_m3,
            rain: ledger.applied_rain_volume_m3,
            losses,
            adjustment: ledger.soil_water_adjustment_volume_m3,
            delivered,
            bound: ledger.residual_m3.abs()
                + bounds.combined_error_m3
                + ledger.transfer_uncertainty_charge_m3
                + export_bound
                + receipt_bound,
            magnitude: ledger.initial_storage_m3
                + ledger.storage_m3
                + ledger.applied_rain_volume_m3
                + losses
                + ledger.soil_water_adjustment_volume_m3.abs()
                + outlet
                + 2.0 * delivered,
        }))
    }
}

/// Runoff one reach recorded on its runoff bindings, each `[bindings]`.
struct ReachRunoff<R: Runtime<DType = DType>> {
    lifetime: Tensor<R>,
    bound: Tensor<R>,
    /// Last-interval volumes, or `None` before the first interval.
    interval: Option<Tensor<R>>,
}

#[cfg(test)]
mod tests {
    use super::super::basin::{
        Basin, CELL_AREA, Receiver, advance, alpha, balanced, pond_volume, reach_volume,
    };
    use super::super::rain_rows::{ALPHA, FINEST_RELATIVE, RAIN, RAIN_SHA256, ROWS, STEPS};
    use super::super::scenario::share;
    use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult};

    /// Recession exponent of the reservoir law.
    const EXPONENT: f64 = 1.6666667;
    /// Horizon of every refinement run, in s.
    const HORIZON: f64 = 4.0;

    /// Water the receiver of `batch` holds above its start.
    type Received = fn(&Batch) -> TestResult<f64>;

    /// Runs `basin` over `HORIZON` in intervals of `dt` with `rain` m/s and returns the water
    /// its receiver gained at each whole second.
    ///
    /// Every interval keeps its full length, passes the composite balance, and holds the mass
    /// identity: receiver gain plus runoff storage equals initial runoff storage plus rain.
    fn run(basin: &Basin, dt: f64, rain: f64, received: Received) -> TestResult<Vec<f64>> {
        let mut basin = basin.clone();
        basin.maximum_step_s = dt;
        let mut batch = basin.build()?;
        let start = received(&batch)?;
        let initial = basin.depth_m * CELL_AREA;
        let mut gains = Vec::new();
        for _ in 0..(HORIZON / dt) as usize {
            let end = batch.time() + dt;
            batch = advance(&batch, dt, rain)?;
            assert_eq!(batch.time().to_bits(), end.to_bits(), "dt={dt}");
            balanced(&batch)?;
            let runoff = batch.runoff().ok_or("the runoff child")?.ledger();
            let gain = received(&batch)? - start;
            let identity = gain + runoff.storage_m3 - initial - runoff.applied_rain_volume_m3;
            let bound = batch.ledger()?.bound_m3;
            assert!(
                identity.abs() <= bound,
                "identity={identity}, bound={bound}"
            );
            if batch.time().fract() == 0.0 {
                gains.push(gain);
            }
        }
        Ok(gains)
    }

    /// Exact recession export of the default basin up to `t`, in m³.
    fn recession(t: f64) -> TestResult<f64> {
        let d0 = 0.01_f64;
        let p = EXPONENT;
        let depth = (d0.powf(1.0 - p) + (p - 1.0) * alpha()? * t).powf(1.0 / (1.0 - p));
        Ok(CELL_AREA * (d0 - depth))
    }

    /// Fails unless the largest relative error against `exact` falls at every refinement and
    /// the finest run's relative error at the final row stays within `finest`.
    fn refines(
        basin: &Basin,
        rain: f64,
        received: Received,
        exact: &[f64],
        finest: f64,
    ) -> TestResult {
        let mut previous = f64::INFINITY;
        let mut last = f64::INFINITY;
        for dt in STEPS {
            let gains = run(basin, dt, rain, received)?;
            assert_eq!(gains.len(), exact.len());
            let errors: Vec<f64> = gains
                .iter()
                .zip(exact)
                .map(|(gain, exact)| (gain - exact).abs() / exact)
                .collect();
            let error = errors.iter().copied().fold(0.0, f64::max);
            assert!(
                error < previous,
                "dt={dt}, error={error}, previous={previous}"
            );
            previous = error;
            last = errors.last().copied().unwrap_or(f64::INFINITY);
        }
        assert!(last <= finest, "finest final-row error={last}");
        Ok(())
    }

    /// Zero-rain recession of 1 cm into the closed pond, against
    /// `d(t) = (d0^(1-p) + (p-1) α t)^(1/(1-p))`.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - Every interval keeps its length, balances, and holds the mass identity.
    /// - The largest relative export error over `t = 1, 2, 3, 4` s falls at each of five
    ///   halvings from 1 s.
    /// - The finest run's relative error at `t = 4` s stays within 2e-3.
    #[test]
    fn recession_into_the_pond_converges_to_the_closed_form() -> TestResult {
        let exact = (1..=4)
            .map(|t| recession(f64::from(t)))
            .collect::<TestResult<Vec<_>>>()?;
        refines(&Basin::into(Receiver::Pool), 0.0, pond_volume, &exact, 2e-3)
    }

    /// The same recession into the middle cell of the walled reach.
    ///
    /// Frozen criteria, fixed before the first run: as for the pond.
    #[test]
    fn recession_into_a_reach_converges_to_the_closed_form() -> TestResult {
        let exact = (1..=4)
            .map(|t| recession(f64::from(t)))
            .collect::<TestResult<Vec<_>>>()?;
        let basin = Basin::into(Receiver::Reach(vec![share(1, 1.0)]));
        refines(&basin, 0.0, reach_volume, &exact, 2e-3)
    }

    /// Constant rain on the dry subcatchment into the closed pond, against the 50-digit
    /// reference of `resource/evidence/runoff/constant_rain.py`.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The fixture's source hash and the model coefficient match the reference inputs.
    /// - Every interval keeps its length, balances, and holds the mass identity.
    /// - The largest relative error over the reference rows falls at each refinement, and
    ///   the finest run's relative error at the final row stays within the reference
    ///   tolerance.
    #[test]
    fn constant_rain_into_the_pond_matches_the_reference() -> TestResult {
        assert_eq!(
            RAIN_SHA256,
            "36db13b34d4e5805ebe10144eb0651be800e87e727972d81250c6f24e47cfe40"
        );
        assert_eq!(alpha()?.to_bits(), ALPHA);
        let exact: Vec<f64> = ROWS
            .iter()
            .map(|&(_, volume)| f64::from_bits(volume))
            .collect();
        for (index, &(time, _)) in ROWS.iter().enumerate() {
            assert_eq!(f64::from_bits(time), (index + 1) as f64);
        }
        let mut basin = Basin::into(Receiver::Pool);
        basin.depth_m = 0.0;
        refines(
            &basin,
            f64::from_bits(RAIN),
            pond_volume,
            &exact,
            FINEST_RELATIVE,
        )
    }
}
