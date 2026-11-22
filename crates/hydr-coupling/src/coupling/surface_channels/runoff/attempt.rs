//! The runoff trial of one attempt and its delivery volumes.
//!
//! Each attempt trials the runoff child from its accepted state to the attempt end with the
//! admitted rates. A shorter attempt recomputes runoff. It never scales a longer volume.
//!
//! Delivery lane `j` of outlet entry `i` takes `D_j = O_i · ŵ_j` of the child export `O`. A
//! weight of exactly one keeps `D_j` equal to `O_i` in every bit. The parts of an entry with
//! `n` targets sum to `O_i` within `γ_n |O_i|`, with `γ_n = nε / (1 − nε)`.

use super::admission::{CheckedRunoff, plan_error};
use super::topology::RunoffTopology;
use crate::channel::ChannelTensorLateralInput;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::context::{Context, Ops};
use crate::runoff::{AdmittedRates, SubcatchmentBatch};
use numr::dtype::DType;
use numr::ops::{BinaryOps, IndexingOps, ShapeOps};
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Runoff operands of one attempt.
pub(in super::super) struct RunoffAttempt<R: Runtime<DType = DType>> {
    /// Runoff child at the attempt end.
    pub next: SubcatchmentBatch<R>,
    /// Delivered volume of every delivery lane, `[N]`.
    pub delivered: Tensor<R>,
    /// Host copy of `delivered`, in m³.
    pub delivered_m3: Vec<f64>,
    /// Runoff inflow inputs of every setup reach, in binding order.
    pub laterals: Vec<Vec<ChannelTensorLateralInput<R>>>,
    /// Delivered volume of every runoff pool binding, `[Np]`, or `None` without pool lanes.
    pub pool_volumes: Option<Tensor<R>>,
    /// Upward sum of every outlet entry's split bound, in m³. Zero without spans.
    pub split_bound_m3: f64,
    /// Runoff inflow of every surface exchange lane, `[S + R]`, or `None` without surface
    /// lanes.
    pub surface: Option<Tensor<R>>,
    /// Surface runoff of this interval: the delivered volume and the lane summation bound, in m³.
    /// Both zero without surface lanes.
    pub surface_step: SurfaceRunoffStep,
}

/// Runoff the surface takes over one interval, in m³.
#[derive(Clone, Copy, Debug, Default)]
pub(in super::super) struct SurfaceRunoffStep {
    /// Delivered volume of every surface delivery lane, summed on the host.
    pub volume_m3: f64,
    /// Upper bound on the summation error of the exchange-lane sums.
    pub bound_m3: f64,
}

/// Upper bound on `|Σ parts − O|` of an entry with `n` targets and export `o`, in m³.
pub(super) fn split_bound(n: usize, o: f64) -> f64 {
    if n <= 1 || o == 0.0 {
        return 0.0;
    }
    let width = n as f64 * f64::EPSILON;
    (width * o.abs() / (1.0 - width)).next_up()
}

impl<R: Runtime<DType = DType>> Context<R>
where
    R::Client: Ops<R>,
{
    /// Runoff topology of this coordinator.
    pub fn runoff_topology(&self) -> Result<&RunoffTopology<R>, SurfaceChannelError> {
        self.runoff
            .as_ref()
            .ok_or_else(|| SurfaceChannelError::Numerical("runoff operands are missing".into()))
    }

    /// Trials `child` to `end` under `rates` and builds the delivery operands.
    ///
    /// Rejects an export that is negative or nonfinite, an entry without lanes that exports
    /// water, a single-target part that differs from its entry in any bit, and span parts
    /// outside their split bound. Each such rejection is numerical.
    pub fn runoff_attempt(
        &self,
        runoff: &CheckedRunoff,
        rates: &AdmittedRates,
        child: &SubcatchmentBatch<R>,
        end: f64,
    ) -> Result<RunoffAttempt<R>, SurfaceChannelError> {
        let c = &self.client;
        let topology = self.runoff_topology()?;
        let trial = runoff
            .plan
            .trial_rates_to(rates, Some(child), end)
            .map_err(|error| plan_error("runoff-trial", error))?
            .ok_or_else(|| SurfaceChannelError::Numerical("the runoff trial is missing".into()))?;
        let outlet = trial
            .original_outlet()
            .map_err(SurfaceChannelError::Runoff)?
            .contiguous()?;
        if outlet.shape() != [topology.entries.len()] {
            return Err(SurfaceChannelError::Numerical(format!(
                "runoff export shape={:?}, entries={}: the runoff child differs from its plan",
                outlet.shape(),
                topology.entries.len()
            )));
        }
        let gathered = c.index_select(&outlet, 0, &topology.gather)?;
        let delivered = match &topology.weights {
            Some(weights) => c.mul(&gathered, weights)?,
            None => gathered,
        };
        let host = c.cat(&[&outlet, &delivered], 0)?.try_to_vec::<f64>()?;
        let (exports, parts) = host.split_at(topology.entries.len());
        let split_bound_m3 = check_parts(topology, exports, parts)?;
        let laterals = (0..topology.reaches.len())
            .map(|reach| topology.laterals(reach, |lane| Ok(delivered.narrow(0, lane, 1)?)))
            .collect::<Result<Vec<_>, _>>()?;
        let pool_volumes = topology
            .pool_gather
            .as_ref()
            .map(|gather| c.index_select(&delivered, 0, gather))
            .transpose()?;
        let (surface, surface_step) = match &topology.surface {
            Some(lanes) => {
                let volume_m3 = lanes
                    .lane_of
                    .iter()
                    .zip(parts)
                    .filter_map(|(lane, part)| lane.map(|_| part))
                    .sum();
                let step = SurfaceRunoffStep {
                    volume_m3,
                    bound_m3: lanes.sum_bound(parts),
                };
                (Some(lanes.lane_inflow(c, &delivered)?), step)
            }
            None => (None, SurfaceRunoffStep::default()),
        };
        Ok(RunoffAttempt {
            next: trial.into_next(),
            delivered_m3: parts.to_vec(),
            delivered,
            laterals,
            pool_volumes,
            split_bound_m3,
            surface,
            surface_step,
        })
    }
}

/// Checks the delivery `parts` of every outlet entry against its export and returns the summed
/// split bound, in m³.
fn check_parts<R: Runtime<DType = DType>>(
    topology: &RunoffTopology<R>,
    exports: &[f64],
    parts: &[f64],
) -> Result<f64, SurfaceChannelError> {
    let mut total = 0.0_f64;
    for (entry, (&export, lanes)) in exports.iter().zip(&topology.entries).enumerate() {
        if !export.is_finite() || export < 0.0 {
            return Err(SurfaceChannelError::Numerical(format!(
                "runoff export entry={entry} volume={export}: the runoff child exports negative or nonfinite water"
            )));
        }
        let values: Vec<f64> = lanes
            .iter()
            .map(|&lane| parts.get(lane).copied().unwrap_or(f64::NAN))
            .collect();
        match values.as_slice() {
            [] if export != 0.0 => {
                return Err(SurfaceChannelError::Numerical(format!(
                    "runoff export entry={entry} volume={export}: an absent or fully routed subarea exports unbound water"
                )));
            }
            [] => {}
            [single] if single.to_bits() != export.to_bits() => {
                return Err(SurfaceChannelError::Numerical(format!(
                    "runoff delivery entry={entry} part={single}, export={export}: a single target differs from its export"
                )));
            }
            [_] => {}
            many => {
                let sum: f64 = many.iter().sum();
                let bound = split_bound(many.len(), export);
                if (sum - export).abs() > bound {
                    return Err(SurfaceChannelError::Numerical(format!(
                        "runoff delivery entry={entry} parts sum={sum}, export={export}, bound={bound}: the span parts depart from their export"
                    )));
                }
                total = (total + bound).next_up();
            }
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::super::basin::{
        Basin, REACH, Receiver, advance, balanced, forcing, isolated, transfers,
    };
    use super::super::scenario::share;
    use super::split_bound;
    use crate::channel::{ChannelLateralBinding, ChannelLateralDirection, ChannelLateralInput};
    use crate::coupling::surface_channels::SurfaceChannelBatch;
    use crate::coupling::surface_channels::structures::scenario::{Batch, TestResult, client};
    use crate::forcing::WaterAmount;

    #[test]
    fn the_split_bound_vanishes_for_single_targets_and_grows_with_the_span() {
        assert_eq!(split_bound(1, 5.0).to_bits(), 0.0_f64.to_bits());
        assert_eq!(split_bound(3, 0.0).to_bits(), 0.0_f64.to_bits());
        let two = split_bound(2, 1.0);
        let four = split_bound(4, 1.0);
        assert!((2.0 * f64::EPSILON..3.0 * f64::EPSILON).contains(&two));
        assert!(four > two);
    }

    /// A deep reach limits the interval by its CFL number, after the runoff admits its rates.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The accepted end lies before the requested end.
    /// - The delivered volume equals the runoff child trialled alone to the accepted end, bit
    ///   for bit, and the pond holds those bits.
    /// - It differs from the full-interval volume scaled by the duration ratio.
    #[test]
    fn a_shortened_attempt_recomputes_runoff_from_the_accepted_state() -> TestResult {
        let mut basin = Basin::into(Receiver::Pool);
        basin.reach_stage_m = 0.5;
        let batch = basin.build()?;
        let next = advance(&batch, 0.5, 0.0)?;
        let end = next.time();
        assert!(end < 0.5, "end={end}");
        let transfers = transfers(&next)?;
        let [transfer] = transfers.as_slice() else {
            return Err("one delivery lane".into());
        };
        let alone = isolated(&batch, end)?;
        assert_eq!(transfer.volume_m3.to_bits(), alone[0].to_bits());
        let pond = next.pools().ok_or("the pond")?.states()?[0].volume_m3;
        assert_eq!(pond.to_bits(), transfer.volume_m3.to_bits());
        let scaled = isolated(&batch, 0.5)?[0] * (end / 0.5);
        assert!(
            (scaled - transfer.volume_m3).abs() > 1e-3 * transfer.volume_m3,
            "scaled={scaled}, delivered={}",
            transfer.volume_m3
        );
        balanced(&next)
    }

    /// One outlet spans reach cells 0 and 2 with weights 1 and 3.
    ///
    /// Frozen criteria, fixed before the first run:
    /// - The two parts sum to the export within the split bound and keep the 1:3 ratio within
    ///   four ulps.
    /// - Each lane ledger holds its part after the first interval.
    /// - Every interval of four balances.
    #[test]
    fn span_parts_sum_to_their_export_within_the_split_bound() -> TestResult {
        let basin = Basin::into(Receiver::Reach(vec![share(0, 1.0), share(2, 3.0)]));
        let batch = basin.build()?;
        let mut next = advance(&batch, 0.25, 0.0)?;
        let export = isolated(&batch, 0.25)?[0];
        let parts: Vec<f64> = transfers(&next)?
            .iter()
            .map(|transfer| transfer.volume_m3)
            .collect();
        let [low, high] = parts.as_slice() else {
            return Err("two delivery lanes".into());
        };
        assert!((low + high - export).abs() <= split_bound(2, export));
        assert!(
            (high / low - 3.0).abs() <= 4.0 * 3.0 * f64::EPSILON,
            "{parts:?}"
        );
        let ledgers = next.runoff_ledgers()?;
        for (ledger, part) in ledgers.iter().zip(&parts) {
            assert_eq!(ledger.delivered_volume_m3.to_bits(), part.to_bits());
        }
        for _ in 0..3 {
            next = advance(&next, 0.25, 0.0)?;
            balanced(&next)?;
        }
        Ok(())
    }

    /// Runoff into a moving reach cell matches a caller inflow of the same interval volume with
    /// zero velocity.
    ///
    /// Frozen criteria, fixed before the first run: every reach cell's volume and momentum, and
    /// the pond volume, agree bit for bit after one interval.
    #[test]
    fn runoff_enters_a_moving_reach_without_momentum() -> TestResult {
        let mut basin = Basin::into(Receiver::Reach(vec![share(1, 1.0)]));
        basin.reach_momentum = 0.05;
        let batch = basin.build()?;
        let next = advance(&batch, 0.25, 0.0)?;
        assert_eq!(next.time().to_bits(), 0.25_f64.to_bits());
        let volume = transfers(&next)?
            .first()
            .ok_or("one delivery lane")?
            .volume_m3;
        let (mut setup, mut initial) = basin.parts()?;
        setup.runoff = None;
        initial.runoff.clear();
        setup.reaches[0]
            .definition
            .lateral_bindings
            .push(ChannelLateralBinding {
                boundary_id: "caller:runoff".into(),
                cell_id: "cell-1".into(),
                direction: ChannelLateralDirection::Inflow,
                ownership_assertion: "The caller holds this water".into(),
            });
        let caller = SurfaceChannelBatch::new(client(), setup, initial)?;
        let mut held = forcing(&caller, 0.25, 0.0);
        held.runoff.clear();
        held.reaches[0].lateral_inputs = vec![ChannelLateralInput::Inflow {
            boundary_id: "caller:runoff".into(),
            amount: WaterAmount::IntervalVolume { volume_m3: volume },
            longitudinal_velocity_m_per_s: 0.0,
        }];
        let reference = caller.trial_to(0.25, &held)?.into_next();
        let cells = |batch: &Batch| -> TestResult<Vec<(u64, u64)>> {
            Ok(batch
                .reach(REACH)
                .ok_or("the reach")?
                .state()?
                .iter()
                .map(|cell| {
                    (
                        cell.volume_m3.to_bits(),
                        cell.longitudinal_momentum_m4_per_s.to_bits(),
                    )
                })
                .collect())
        };
        assert_eq!(cells(&next)?, cells(&reference)?);
        let pond = |batch: &Batch| -> TestResult<u64> {
            Ok(batch.pools().ok_or("the pond")?.states()?[0]
                .volume_m3
                .to_bits())
        };
        assert_eq!(pond(&next)?, pond(&reference)?);
        Ok(())
    }
}
