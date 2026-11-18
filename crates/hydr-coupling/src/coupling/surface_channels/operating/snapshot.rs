//! Caller-owned packages of the operating children and their checked restoration.

use super::{OperatingChildren, child_error};
use crate::coupling::checkpoint;
use crate::coupling::surface_channels::SurfaceChannelError;
use crate::structures::{ControlledGateSnapshotData, ControlledPumpSnapshotData};
use numr::dtype::DType;
use numr::ops::{
    BinaryOps, CompareOps, ConditionalOps, ReduceOps, ScalarOps, TwoSumOps, TypeConversionOps,
    UnaryOps,
};
use numr::runtime::Runtime;

/// Complete packages of the stateful operating children at one accepted time.
///
/// A prescribed trajectory holds no state, so the setup alone restores it.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceChannelOperatingData {
    /// Controlled-gate package, or `None` without controlled gates.
    pub gate: Option<ControlledGateSnapshotData>,
    /// Controlled-pump package, or `None` without controlled pumps.
    pub pump: Option<ControlledPumpSnapshotData>,
}

impl SurfaceChannelOperatingData {
    /// Checks every nested package identity. Errors start with `path`.
    pub(in super::super) fn check_identities(&self, path: &str) -> Result<(), String> {
        if let Some(gate) = &self.gate {
            checkpoint::controlled_gate(&format!("{path}.gate"), gate)?;
        }
        if let Some(pump) = &self.pump {
            checkpoint::controlled_pump(&format!("{path}.pump"), pump)?;
        }
        Ok(())
    }
}

impl<R: Runtime<DType = DType>> OperatingChildren<R> {
    /// Projects the package of every stateful child.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] naming the child for every projection error.
    pub fn data(&self) -> Result<SurfaceChannelOperatingData, SurfaceChannelError> {
        Ok(SurfaceChannelOperatingData {
            gate: self
                .gate
                .as_ref()
                .map(|gate| gate.snapshot_data())
                .transpose()
                .map_err(child_error("controlled-gate"))?,
            pump: self
                .pump
                .as_ref()
                .map(|pump| pump.snapshot_data())
                .transpose()
                .map_err(child_error("controlled-pump"))?,
        })
    }

    /// Accepted time and interval count of every stateful child, as `(child, time, count)`.
    pub fn clocks(&self) -> Vec<(&'static str, f64, usize)> {
        let mut clocks = Vec::new();
        if let Some(gate) = &self.gate {
            clocks.push(("operating.gate", gate.time(), gate.accepted_steps()));
            clocks.push((
                "operating.gate.actuators",
                gate.actuators().time(),
                gate.actuators().accepted_steps(),
            ));
        }
        if let Some(pump) = &self.pump {
            clocks.push(("operating.pump", pump.time(), pump.accepted_steps()));
        }
        clocks
    }
}

impl<R: Runtime<DType = DType>> OperatingChildren<R>
where
    R::Client: BinaryOps<R>
        + CompareOps<R>
        + ConditionalOps<R>
        + ReduceOps<R>
        + ScalarOps<R>
        + TwoSumOps<R>
        + TypeConversionOps<R>
        + UnaryOps<R>,
{
    /// These children with every stateful child restored from `data`.
    ///
    /// The binding stays: each child checks its own immutable model against the package.
    ///
    /// # Errors
    /// Returns [`SurfaceChannelError::Input`] naming `path` for a package whose child presence
    /// differs, and naming the child for every child restoration rejection.
    pub fn restored(
        &self,
        path: &str,
        data: &SurfaceChannelOperatingData,
    ) -> Result<Self, SurfaceChannelError> {
        let mismatch = |child: &str| {
            SurfaceChannelError::input(format!(
                "checkpoint {path}.{child} presence differs from the destination operating children: supply an unchanged complete same-build package"
            ))
        };
        let gate = match (&self.gate, &data.gate) {
            (Some(gate), Some(package)) => {
                let mut gate = gate.clone();
                gate.restore_data(package)
                    .map_err(child_error("controlled-gate"))?;
                Some(gate)
            }
            (None, None) => None,
            _ => return Err(mismatch("gate")),
        };
        let pump = match (&self.pump, &data.pump) {
            (Some(pump), Some(package)) => {
                let mut pump = pump.clone();
                pump.restore_data(package)
                    .map_err(child_error("controlled-pump"))?;
                Some(pump)
            }
            (None, None) => None,
            _ => return Err(mismatch("pump")),
        };
        Ok(Self {
            binding: std::sync::Arc::clone(&self.binding),
            gate,
            pump,
        })
    }
}
