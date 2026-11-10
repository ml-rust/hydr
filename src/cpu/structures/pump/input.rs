//! Explicit operation admission and IEEE power-of-two proof metadata.

use crate::cpu::structures::operations::CheckedOperations;
use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::{StructureError, StructureOperatingPoint};
use numr::ops::ConditionalOps;

#[derive(Debug)]
pub(crate) struct CpuPumpInput {
    pub running: CpuTensor,
    pub isolated: CpuTensor,
    pub speed: CpuTensor,
    pub power_of_two: CpuTensor,
}
impl CpuPumpInput {
    pub(crate) fn new(
        context: &CpuStructureContext,
        operations: &CheckedOperations<'_>,
    ) -> Result<Option<Self>, StructureError> {
        if context.pump.is_none() {
            return Ok(None);
        }
        let n = operations.len();
        let (mut running, mut isolated, mut speed, mut power) =
            (vec![0.0; n], vec![0.0; n], vec![1.0; n], vec![1.0; n]);
        for index in 0..n {
            match operations.supplied(index) {
                Some(StructureOperatingPoint::PumpStoppedIsolated) => isolated[index] = 1.0,
                Some(StructureOperatingPoint::PumpRunning { speed_ratio }) => {
                    running[index] = 1.0;
                    speed[index] = speed_ratio;
                    power[index] = if super::super::power_metadata::power_of_two(speed_ratio) {
                        1.0
                    } else {
                        0.0
                    };
                }
                Some(
                    StructureOperatingPoint::Passive | StructureOperatingPoint::GateOpening { .. },
                )
                | None => {}
            }
        }
        let mut input = Self {
            running: context.tensor(&running)?,
            isolated: context.tensor(&isolated)?,
            speed: context.tensor(&speed)?,
            power_of_two: context.tensor(&power)?,
        };
        if let Some(derived) = operations.derived_pump() {
            let c = &context.client;
            input.running = c.where_cond(derived.mask, derived.running, &input.running)?;
            input.isolated = c.where_cond(derived.mask, derived.isolated, &input.isolated)?;
            input.speed = c.where_cond(derived.mask, derived.speed, &input.speed)?;
            input.power_of_two =
                c.where_cond(derived.mask, derived.power_of_two, &input.power_of_two)?;
        }
        Ok(Some(input))
    }
}
