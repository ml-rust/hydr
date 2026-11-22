//! Host receipts of runoff delivery: interval transfers and lifetime lane ledgers.

use super::topology::RunoffTopology;
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelError, SurfaceChannelRunoffLedger,
    SurfaceChannelRunoffTransfer,
};
use numr::dtype::DType;
use numr::runtime::Runtime;

/// Runoff delivery of one accepted interval, in delivery-lane order.
#[derive(Clone, Debug)]
pub(in super::super) struct RunoffStep {
    /// Delivered volume of every lane, in m³.
    pub delivered_m3: Vec<f64>,
    /// Receiving stage of every lane at the accepted start, in metres.
    pub start_stage_m: Vec<f64>,
    /// Receiving stage of every lane at the accepted end, in metres.
    pub end_stage_m: Vec<f64>,
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R> {
    /// Runoff transfer of every delivery lane over the last accepted interval.
    ///
    /// Empty without runoff or before the first accepted interval.
    #[doc(hidden)]
    pub fn runoff_transfers(&self) -> Vec<SurfaceChannelRunoffTransfer> {
        let (Some(topology), Some(step)) = (
            self.context.runoff.as_ref(),
            self.last.as_ref().and_then(|last| last.runoff.as_ref()),
        ) else {
            return Vec::new();
        };
        (0..topology.lane_count())
            .map(|lane| SurfaceChannelRunoffTransfer {
                outlet_id: topology.outlets[lane].clone(),
                subarea: topology.subareas[lane],
                target: topology.nodes[lane].clone(),
                volume_m3: value(&step.delivered_m3, lane),
                start_stage_m: value(&step.start_stage_m, lane),
                end_stage_m: value(&step.end_stage_m, lane),
                invert_elevation_m: topology.inverts[lane],
            })
            .collect()
    }

    /// Lifetime delivered volume of every delivery lane. Empty without runoff.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    #[doc(hidden)]
    pub fn runoff_ledgers(&self) -> Result<Vec<SurfaceChannelRunoffLedger>, SurfaceChannelError> {
        let (Some(topology), Some(counters)) =
            (self.context.runoff.as_ref(), self.counters.runoff.as_ref())
        else {
            return Ok(Vec::new());
        };
        let delivered = counters.delivered_m3.try_to_vec::<f64>()?;
        let bounds = counters.roundoff_bound_m3.try_to_vec::<f64>()?;
        Ok(ledgers(topology, &delivered, &bounds))
    }
}

/// Entry `lane` of `values`, NaN when absent.
fn value(values: &[f64], lane: usize) -> f64 {
    values.get(lane).copied().unwrap_or(f64::NAN)
}

/// Lane ledgers of `topology` from host counters.
fn ledgers<R: Runtime<DType = DType>>(
    topology: &RunoffTopology<R>,
    delivered: &[f64],
    bounds: &[f64],
) -> Vec<SurfaceChannelRunoffLedger> {
    (0..topology.lane_count())
        .map(|lane| SurfaceChannelRunoffLedger {
            outlet_id: topology.outlets[lane].clone(),
            subarea: topology.subareas[lane],
            target: topology.nodes[lane].clone(),
            delivered_volume_m3: value(delivered, lane),
            roundoff_bound_m3: value(bounds, lane),
        })
        .collect()
}
