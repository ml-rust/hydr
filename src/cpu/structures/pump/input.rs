//! Explicit operation admission and IEEE power-of-two proof metadata.

use crate::cpu::{CpuTensor, structures::CpuStructureContext};
use crate::structures::{
    StructureError, StructureInputErrorKind, StructureLaw, StructureOperatingPoint,
    StructureOperation,
};

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
        operations: &[StructureOperation],
    ) -> Result<Option<Self>, StructureError> {
        if operations.len() != context.elements.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                "supply exactly one ordered identity-bound operation per structure",
            ));
        }
        let n = operations.len();
        let (mut running, mut isolated, mut speed, mut power) =
            (vec![0.0; n], vec![0.0; n], vec![1.0; n], vec![1.0; n]);
        for (index, (operation, element)) in operations.iter().zip(&context.elements).enumerate() {
            if operation.structure_id != element.id {
                return Err(StructureError::input(
                    StructureInputErrorKind::Identity,
                    format!(
                        "operation={:?},expected={:?}: match complete original structure order",
                        operation.structure_id, element.id
                    ),
                )
                .at(index));
            }
            match (&element.law, operation.operating_point) {
                (StructureLaw::Pump(_), StructureOperatingPoint::PumpStoppedIsolated) => {
                    isolated[index] = 1.0
                }
                (StructureLaw::Pump(_), StructureOperatingPoint::PumpRunning { speed_ratio }) => {
                    if !speed_ratio.is_finite() || speed_ratio <= 0.0 {
                        return Err(StructureError::input(StructureInputErrorKind::OperatingPoint,format!("speed_ratio={speed_ratio}: supply positive finite speed or explicit stopped-isolated operation")).at(index));
                    }
                    running[index] = 1.0;
                    speed[index] = speed_ratio;
                    power[index] = if power_of_two(speed_ratio) { 1.0 } else { 0.0 };
                }
                (StructureLaw::Pump(_), _) => {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "pump requires explicit running or stopped-isolated operation",
                    )
                    .at(index));
                }
                (_, StructureOperatingPoint::Passive) => {}
                _ => {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "passive law requires an explicit Passive operation",
                    )
                    .at(index));
                }
            }
        }
        if context.pump.is_none() {
            return Ok(None);
        }
        Ok(Some(Self {
            running: context.tensor(&running)?,
            isolated: context.tensor(&isolated)?,
            speed: context.tensor(&speed)?,
            power_of_two: context.tensor(&power)?,
        }))
    }
}
fn power_of_two(value: f64) -> bool {
    let bits = value.to_bits();
    let exponent = (bits >> 52) & 0x7ff;
    let fraction = bits & ((1u64 << 52) - 1);
    if exponent == 0 {
        fraction != 0 && (fraction & (fraction - 1)) == 0
    } else {
        fraction == 0
    }
}
