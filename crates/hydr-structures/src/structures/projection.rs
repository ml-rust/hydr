//! Explicit proposal projection without numerical host-state advancement.
use numr::{dtype::DType, runtime::Runtime};

use super::proposal::ProposalData;
use super::{
    FlowDirection, StructureError, StructureHydraulicOperands, StructureLaw, StructureLawKind,
    StructureOutput, StructureRegime, StructureStatus,
};
use crate::structures::tensor::codes;
use numr::tensor::Tensor;

pub(super) fn read<R: Runtime<DType = DType>>(
    tensor: &Tensor<R>,
    n: usize,
) -> Result<Vec<f64>, StructureError> {
    if tensor.shape() != [n] {
        return Err(StructureError::numerical(format!(
            "projection shape={:?}: require {n} ordered structure values",
            tensor.shape()
        )));
    }
    Ok(tensor.try_to_vec::<f64>()?)
}

pub(super) fn outputs<R: Runtime<DType = DType>>(
    data: &ProposalData<R>,
) -> Result<Vec<StructureOutput>, StructureError> {
    let operations = super::operation_projection::project(&data.operations)?;
    let n = data.context.elements.len();
    let r = &data.result;
    let discharge = read(&r.discharge, n)?;
    let regimes = read(&r.regime, n)?;
    let statuses = read(&r.status, n)?;
    let directions = read(&r.direction, n)?;
    let gap = read(&r.head_gap, n)?;
    let hu = read(&r.upstream_head, n)?;
    let hd = read(&r.downstream_head, n)?;
    let ratio = read(&r.ratio, n)?;
    let k = read(&r.correction_k, n)?;
    let logarithm = read(&r.log_magnitude, n)?;
    let pump = r
        .pump
        .as_ref()
        .map(|r| super::pump::PumpOperandProjection::new(r, n))
        .transpose()?;
    let gate = r
        .gate
        .as_ref()
        .map(|result| super::gate::GateOperandProjection::new(result, n))
        .transpose()?;
    let culvert = r
        .culvert
        .as_ref()
        .map(|result| -> Result<[Vec<f64>; 4], StructureError> {
            Ok([
                read(&result.headwater, n)?,
                read(&result.tailwater, n)?,
                read(&result.lower, n)?,
                read(&result.upper, n)?,
            ])
        })
        .transpose()?;
    (0..n)
        .map(|element| {
            let definition = &data.context.elements[element];
            let regime = regime(regimes[element]).map_err(|e| e.at(element))?;
            let status = status(statuses[element]).map_err(|e| e.at(element))?;
            let direction = if directions[element] == 1.0 {
                Some(FlowDirection::AtoB)
            } else if directions[element] == -1.0 {
                Some(FlowDirection::BtoA)
            } else if directions[element] == 0.0 {
                None
            } else {
                return Err(
                    StructureError::numerical("invalid internal proposal direction code")
                        .at(element),
                );
            };
            let hydraulic_operands = if status == StructureStatus::PumpRunning
                || status == StructureStatus::PumpShutoff
            {
                let operands = pump.as_ref().ok_or_else(|| {
                    StructureError::numerical("evaluated pump output lacks normalization operands")
                        .at(element)
                })?;
                Some(StructureHydraulicOperands::Pump(
                    operands.at(element, operations[element].operating_point)?,
                ))
            } else if status == StructureStatus::Active {
                Some(match regime {
                    StructureRegime::FreeGateAperture
                    | StructureRegime::MixedGateAperture
                    | StructureRegime::SubmergedGateAperture => {
                        let operands = gate.as_ref().ok_or_else(|| {
                            StructureError::numerical(
                                "evaluated gate output lacks geometry operands",
                            )
                            .at(element)
                        })?;
                        StructureHydraulicOperands::HydrostaticRectangularGate(
                            operands.at(element, operations[element].operating_point)?,
                        )
                    }
                    StructureRegime::CulvertInletControl
                    | StructureRegime::CulvertOutletControl
                    | StructureRegime::CulvertControlTie => {
                        let [headwater, tailwater, lower, upper] =
                            culvert.as_ref().ok_or_else(|| {
                                StructureError::numerical(
                                    "evaluated culvert output lacks its control operands",
                                )
                                .at(element)
                            })?;
                        StructureHydraulicOperands::Culvert {
                            headwater_m: headwater[element],
                            tailwater_m: tailwater[element],
                            barrel_discharge_m3_per_s: (lower[element], upper[element]),
                        }
                    }
                    StructureRegime::SubmergedOrifice => {
                        StructureHydraulicOperands::SubmergedOrifice {
                            head_gap_m: gap[element],
                            log_discharge_magnitude: logarithm[element],
                        }
                    }
                    StructureRegime::FreeWeirOverflow => StructureHydraulicOperands::FreeWeir {
                        upstream_head_m: hu[element],
                        log_discharge_magnitude: logarithm[element],
                    },
                    StructureRegime::SubmergedWeirOverflow => {
                        StructureHydraulicOperands::SubmergedWeir {
                            upstream_head_m: hu[element],
                            downstream_head_m: hd[element],
                            head_gap_m: gap[element],
                            submergence_ratio: ratio[element],
                            correction_k: k[element],
                            log_discharge_magnitude: logarithm[element],
                        }
                    }
                    _ => {
                        return Err(StructureError::numerical(
                            "physical-zero regime cannot carry active hydraulic operands",
                        )
                        .at(element));
                    }
                })
            } else {
                None
            };
            Ok(StructureOutput {
                operation: operations[element].operating_point,
                element,
                input: data.heads[element].clone(),
                law: match &definition.law {
                    StructureLaw::Culvert(_) => StructureLawKind::Culvert,
                    StructureLaw::HydrostaticRectangularGate(_) => {
                        StructureLawKind::HydrostaticRectangularGate
                    }
                    StructureLaw::Pump(_) => StructureLawKind::Pump,
                    StructureLaw::SubmergedOrifice(_) => StructureLawKind::SubmergedOrifice,
                    StructureLaw::SuppressedRectangularWeir(_) => {
                        StructureLawKind::SuppressedRectangularWeir
                    }
                },
                direction_policy: definition.direction,
                regime,
                status,
                proposed_direction: direction,
                discharge_m3_per_s: discharge[element],
                hydraulic_operands,
            })
        })
        .collect()
}
fn regime(value: f64) -> Result<StructureRegime, StructureError> {
    if value == codes::GATE_CLOSED {
        Ok(StructureRegime::GateClosed)
    } else if value == codes::GATE_BELOW_SILL {
        Ok(StructureRegime::GateBelowSill)
    } else if value == codes::FREE_GATE {
        Ok(StructureRegime::FreeGateAperture)
    } else if value == codes::MIXED_GATE {
        Ok(StructureRegime::MixedGateAperture)
    } else if value == codes::SUBMERGED_GATE {
        Ok(StructureRegime::SubmergedGateAperture)
    } else if value == codes::PUMP_CURVE {
        Ok(StructureRegime::PumpCurve)
    } else if value == codes::PUMP_ISOLATED {
        Ok(StructureRegime::PumpIsolated)
    } else if value == codes::ORIFICE {
        Ok(StructureRegime::SubmergedOrifice)
    } else if value == codes::FREE_WEIR {
        Ok(StructureRegime::FreeWeirOverflow)
    } else if value == codes::SUBMERGED_WEIR {
        Ok(StructureRegime::SubmergedWeirOverflow)
    } else if value == codes::BELOW_CREST {
        Ok(StructureRegime::BelowCrest)
    } else if value == codes::EQUAL_HEADS {
        Ok(StructureRegime::EqualHeads)
    } else if value == codes::CULVERT_INLET_CONTROL {
        Ok(StructureRegime::CulvertInletControl)
    } else if value == codes::CULVERT_OUTLET_CONTROL {
        Ok(StructureRegime::CulvertOutletControl)
    } else if value == codes::CULVERT_CONTROL_TIE {
        Ok(StructureRegime::CulvertControlTie)
    } else if value == codes::CULVERT_BELOW_INVERT {
        Ok(StructureRegime::CulvertBelowInvert)
    } else if value == codes::CULVERT_UNDECLARED_END {
        Ok(StructureRegime::CulvertUndeclaredEnd)
    } else {
        Err(StructureError::numerical(
            "invalid internal physical-regime code",
        ))
    }
}
fn status(value: f64) -> Result<StructureStatus, StructureError> {
    if value == codes::PUMP_RUNNING {
        Ok(StructureStatus::PumpRunning)
    } else if value == codes::PUMP_SHUTOFF {
        Ok(StructureStatus::PumpShutoff)
    } else if value == codes::PUMP_STOPPED {
        Ok(StructureStatus::PumpStoppedIsolated)
    } else if value == codes::ACTIVE {
        Ok(StructureStatus::Active)
    } else if value == codes::PHYSICAL_ZERO {
        Ok(StructureStatus::PhysicalZero)
    } else if value == codes::BLOCKED {
        Ok(StructureStatus::FlapBlocked)
    } else {
        Err(StructureError::numerical(
            "invalid internal proposal-status code",
        ))
    }
}
