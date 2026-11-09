//! Identity-bound operation admission before any hydraulic-support waiver.

use super::CpuStructureContext;
use crate::structures::{
    StructureError, StructureInputErrorKind, StructureLaw, StructureOperatingPoint,
    StructureOperation,
};

pub(crate) struct CheckedOperations<'a> {
    packets: &'a [StructureOperation],
    isolated: Vec<bool>,
}
impl<'a> CheckedOperations<'a> {
    pub(crate) fn new(
        context: &CpuStructureContext,
        packets: &'a [StructureOperation],
    ) -> Result<Self, StructureError> {
        if packets.len() != context.elements.len() {
            return Err(StructureError::input(
                StructureInputErrorKind::OperatingPoint,
                format!(
                    "operation count={},structure count={}: supply exactly one ordered identity-bound operation per structure",
                    packets.len(),
                    context.elements.len()
                ),
            ));
        }
        let mut isolated = Vec::with_capacity(packets.len());
        for (index, (operation, element)) in packets.iter().zip(&context.elements).enumerate() {
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
            let uninfluential = match (&element.law, operation.operating_point) {
                (StructureLaw::Pump(_), StructureOperatingPoint::PumpStoppedIsolated) => true,
                (StructureLaw::Pump(_), StructureOperatingPoint::PumpRunning { speed_ratio }) => {
                    if !speed_ratio.is_finite() || speed_ratio <= 0.0 {
                        return Err(StructureError::input(StructureInputErrorKind::OperatingPoint,format!("speed_ratio={speed_ratio}: supply positive finite speed or explicit stopped-isolated operation")).at(index));
                    }
                    false
                }
                (
                    StructureLaw::HydrostaticRectangularGate(parameters),
                    StructureOperatingPoint::GateOpening { opening_m },
                ) => {
                    if !opening_m.is_finite()
                        || opening_m < 0.0
                        || opening_m > parameters.maximum_opening_m()
                    {
                        return Err(StructureError::input(
                            StructureInputErrorKind::OperatingPoint,
                            format!(
                                "opening_m={opening_m},maximum_opening_m={}: supply a finite opening in the declared closed range",
                                parameters.maximum_opening_m()
                            ),
                        )
                        .at(index));
                    }
                    opening_m == 0.0
                }
                (
                    StructureLaw::SubmergedOrifice(_) | StructureLaw::SuppressedRectangularWeir(_),
                    StructureOperatingPoint::Passive,
                ) => false,
                (StructureLaw::Pump(_), _) => {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "pump requires explicit running or stopped-isolated operation",
                    )
                    .at(index));
                }
                (StructureLaw::HydrostaticRectangularGate(_), _) => {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "gate requires an explicit identity-bound opening operation",
                    )
                    .at(index));
                }
                _ => {
                    return Err(StructureError::input(
                        StructureInputErrorKind::OperatingPoint,
                        "passive law requires an explicit Passive operation",
                    )
                    .at(index));
                }
            };
            isolated.push(uninfluential);
        }
        Ok(Self { packets, isolated })
    }
    pub(crate) fn packets(&self) -> &[StructureOperation] {
        self.packets
    }
    pub(crate) fn isolated(&self, index: usize) -> bool {
        self.isolated[index]
    }
}
