//! Host receipts of runoff delivery: interval transfers, lifetime lane ledgers, and their
//! checkpoint counters.

use super::counters::RunoffCounters;
use super::topology::RunoffTopology;
use crate::coupling::surface_channels::context::Context;
use crate::coupling::surface_channels::frames::rejected;
use crate::coupling::surface_channels::state::StoredStep;
use crate::coupling::surface_channels::{
    SurfaceChannelBatch, SurfaceChannelError, SurfaceChannelRunoffCounterData,
    SurfaceChannelRunoffLedger, SurfaceChannelRunoffTransfer,
};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient};
use numr::tensor::Tensor;

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

impl<R: Runtime<DType = DType>> StoredStep<R> {
    /// Runoff transfer of every delivery lane of `context` over this interval.
    ///
    /// Empty without runoff.
    pub(in super::super) fn runoff_transfers(
        &self,
        context: &Context<R>,
    ) -> Vec<SurfaceChannelRunoffTransfer> {
        let (Some(topology), Some(step)) = (context.runoff.as_ref(), self.runoff.as_ref()) else {
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
}

impl<R: Runtime<DType = DType>> SurfaceChannelBatch<R> {
    /// Lifetime delivered volume of every delivery lane, in delivery-lane order. Empty without
    /// runoff.
    ///
    /// # Errors
    /// Returns runtime transfer errors.
    pub fn runoff_ledgers(&self) -> Result<Vec<SurfaceChannelRunoffLedger>, SurfaceChannelError> {
        Ok(counter_data(&self.context, self.counters.runoff.as_ref())?.ledgers)
    }
}

/// Host checkpoint counters of `counters`. Empty and zero without runoff.
///
/// # Errors
/// Returns runtime transfer errors.
pub(in super::super) fn counter_data<R: Runtime<DType = DType>>(
    context: &Context<R>,
    counters: Option<&RunoffCounters<R>>,
) -> Result<SurfaceChannelRunoffCounterData, SurfaceChannelError> {
    let (Some(topology), Some(counters)) = (context.runoff.as_ref(), counters) else {
        return Ok(SurfaceChannelRunoffCounterData::default());
    };
    let delivered = counters.delivered_m3.try_to_vec::<f64>()?;
    let bounds = counters.roundoff_bound_m3.try_to_vec::<f64>()?;
    Ok(SurfaceChannelRunoffCounterData {
        ledgers: ledgers(topology, &delivered, &bounds),
        split_bound_m3: counters.split_bound_m3,
    })
}

/// Device counters of `context` from checkpoint counters at `prefix`, or `None` without runoff.
///
/// Each ledger names its lane's outlet, subarea, and target in delivery-lane order. Every value
/// is finite and nonnegative. Before the first interval every value is `+0.0`.
///
/// # Errors
/// Returns [`SurfaceChannelError::Input`] naming the first foreign ledger or value.
pub(in super::super) fn import_counters<R: Runtime<DType = DType>>(
    context: &Context<R>,
    data: &SurfaceChannelRunoffCounterData,
    advanced: bool,
    prefix: &str,
) -> Result<Option<RunoffCounters<R>>, SurfaceChannelError> {
    let path = format!("{prefix}counters.runoff");
    let Some(topology) = context.runoff.as_ref() else {
        if data.ledgers.is_empty() && data.split_bound_m3.to_bits() == 0 {
            return Ok(None);
        }
        return Err(rejected(&path, "holds runoff counters without runoff"));
    };
    let valid = |value: f64| {
        value.is_finite() && value.is_sign_positive() && (advanced || value.to_bits() == 0)
    };
    let invalid =
        "holds a negative or nonfinite value, or a nonzero value before the first interval";
    if !valid(data.split_bound_m3) {
        return Err(rejected(&format!("{path}.split_bound_m3"), invalid));
    }
    let expected = ledgers(topology, &[], &[]);
    let same_lanes = data.ledgers.len() == expected.len()
        && data.ledgers.iter().zip(&expected).all(|(a, b)| {
            a.outlet_id == b.outlet_id && a.subarea == b.subarea && a.target == b.target
        });
    if !same_lanes {
        return Err(rejected(
            &format!("{path}.ledgers"),
            "differs from the delivery lanes in outlet, subarea, target, or order",
        ));
    }
    for (index, ledger) in data.ledgers.iter().enumerate() {
        if !valid(ledger.delivered_volume_m3) || !valid(ledger.roundoff_bound_m3) {
            return Err(rejected(&format!("{path}.ledgers[{index}]"), invalid));
        }
    }
    let device = context.client.device();
    let tensor = |values: Vec<f64>| Tensor::<R>::from_slice(&values, &[values.len()], device);
    Ok(Some(RunoffCounters {
        delivered_m3: tensor(data.ledgers.iter().map(|l| l.delivered_volume_m3).collect())?,
        roundoff_bound_m3: tensor(data.ledgers.iter().map(|l| l.roundoff_bound_m3).collect())?,
        split_bound_m3: data.split_bound_m3,
    }))
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
