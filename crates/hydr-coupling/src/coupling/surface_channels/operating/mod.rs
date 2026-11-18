//! Controlled structures: operating children that set structure operations interval by interval.
//!
//! [`SurfaceChannelOperating`] declares the children as data. Construction builds them fresh: a
//! prescribed gate follows a [`GateTrajectoryBatch`], an actuated gate a [`ControlledGateBatch`],
//! and a controlled pump a [`ControlledPumpBatch`]. Each [`SurfaceChannelSignal`] binds one
//! controller signal to the node whose stage it samples.
//!
//! Each common interval runs these steps:
//! 1. The end is the smallest of the requested end, the start plus `maximum_step_s`, the surface
//!    forcing end, and the next operating event. Events are prescribed knots, actuator events,
//!    control decisions, and the common terminal bound. No interval contains one, and a retry
//!    only shortens the interval.
//! 2. Each attempt prepares every child to its end without samples. A gate takes the midpoint of
//!    its start and end openings, the partner of midpoint heads. A gate closed at both ends takes
//!    exact closure. A pump holds its accepted operation from the start.
//! 3. The fixed point solves the hydraulics under those operations. They depend on no iterate.
//! 4. At a decision end, each signal samples its node's stage in the converged candidate at the
//!    decision time. A dry node rejects. Each child completes with its samples. Its left positions
//!    must equal the prepared bits. The new command acts on the right side only, so no inner
//!    iteration joins controls and hydraulics.
//! 5. Acceptance replaces every child at once, operating children included. A rejected attempt
//!    leaves every child unchanged.

mod admission;
mod batch;
mod decision;
mod declaration;
mod interval;
#[cfg(test)]
mod scenario;
mod signals;
mod snapshot;

pub(super) use admission::{admit, check_manual};
pub(super) use decision::CandidateNodes;
pub(super) use declaration::build_children;
pub use declaration::{
    SurfaceChannelActuators, SurfaceChannelControlledGates, SurfaceChannelControlledPumps,
    SurfaceChannelControls, SurfaceChannelOperating, SurfaceChannelSignal,
};
pub(super) use interval::{AttemptOperation, StepOperation};
pub use snapshot::SurfaceChannelOperatingData;

use crate::structures::{
    ControlledGateBatch, ControlledPumpBatch, GateTrajectoryBatch, OperatingSourceBatch,
    OperatingSourceInterval, OperatingSourceIntervalFrame, StructureError,
};
use numr::dtype::DType;
use numr::runtime::Runtime;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Registry source of the manual structures.
const MANUAL: &str = "manual";
/// Registry source of the trajectory child.
const PRESCRIBED: &str = "prescribed";
/// Registry source of the controlled-gate child.
const ACTUATED: &str = "actuated";
/// Registry source of the controlled-pump child.
const PUMPED: &str = "controlled-pump";

/// Fresh operating children built from a [`SurfaceChannelOperating`] declaration.
///
/// Each child sits on the setup's structure links in ascending identity.
pub(super) struct OperatingBatches<R: Runtime<DType = DType>> {
    /// Gates of every `Actuated` owner, or `None` without one.
    pub gate: Option<ControlledGateBatch<R>>,
    /// Pumps of every `ControlledPump` owner, or `None` without one.
    pub pump: Option<ControlledPumpBatch<R>>,
    /// Gates of every `Prescribed` owner, or `None` without one.
    pub trajectory: Option<GateTrajectoryBatch<R>>,
}

/// Interval operation of one event-free interval from its prepared frames.
///
/// Admission fixes it while the trajectory interpolation bound holds, so a trial needs no
/// further client bound.
type IntervalOperation<R> = fn(
    &OperatingSourceBatch<R>,
    f64,
    f64,
    &[OperatingSourceIntervalFrame<'_, R>],
) -> Result<OperatingSourceInterval<R>, StructureError>;

/// One controller signal with its definition and its donor group.
#[derive(Clone, Debug)]
struct BoundSignal {
    controller_id: String,
    signal_id: String,
    support_id: String,
    datum: String,
    /// `surface-cell:<n>`, `reach:<id>:cell:<k>`, or `pool:<id>`.
    node: String,
    /// Donor group of the node in `S + C + P`.
    group: usize,
}

/// Immutable binding of the operating children to the link table.
struct OperatingBinding<R: Runtime<DType = DType>> {
    registry: OperatingSourceBatch<R>,
    interval: IntervalOperation<R>,
    /// Link table position of every dynamically owned structure.
    links: BTreeMap<String, usize>,
    /// Signals of the controlled-gate child, in its controller order.
    gate_signals: Vec<BoundSignal>,
    /// Signals of the controlled-pump child, in its controller order.
    pump_signals: Vec<BoundSignal>,
}

/// Accepted operating children of a coordinator.
pub(super) struct OperatingChildren<R: Runtime<DType = DType>> {
    binding: Arc<OperatingBinding<R>>,
    gate: Option<ControlledGateBatch<R>>,
    pump: Option<ControlledPumpBatch<R>>,
}

impl<R: Runtime<DType = DType>> Clone for OperatingChildren<R> {
    fn clone(&self) -> Self {
        Self {
            binding: Arc::clone(&self.binding),
            gate: self.gate.clone(),
            pump: self.pump.clone(),
        }
    }
}

impl<R: Runtime<DType = DType>> OperatingChildren<R> {
    /// Accepted controlled-gate child, or `None` without one.
    pub fn gate(&self) -> Option<&ControlledGateBatch<R>> {
        self.gate.as_ref()
    }

    /// Accepted controlled-pump child, or `None` without one.
    pub fn pump(&self) -> Option<&ControlledPumpBatch<R>> {
        self.pump.as_ref()
    }
}

/// A non-retryable rejection of operating child `child`.
fn child_error(child: &'static str) -> impl Fn(StructureError) -> super::SurfaceChannelError {
    move |error| super::SurfaceChannelError::input(format!("operating child={child}: {error}"))
}
