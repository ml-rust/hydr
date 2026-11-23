//! Laws, conditions, head packets and proposals shared by the blockage checks.

use hydr::structures::*;

pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Runtime = crate::runtime::Runtime;

const DATUM: &str = "Analytical blockage datum";
pub const EVIDENCE: &str = "Analytical blockage fixture";

/// Gate opening of every evaluated gate lane, in metres.
pub const OPENING: f64 = 0.5;

/// Weir with crest length 2 m and crest at 1 m.
pub fn weir() -> Result<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(
            2.0,
            1.0,
            1.7,
            WeirApplicability {
                coefficient_provenance: EVIDENCE.into(),
                free_nappe_and_approach_assertion: EVIDENCE.into(),
                submergence_correction_assertion: EVIDENCE.into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Orifice with area 2 m², sill at 0 m and crown at 1 m.
pub fn orifice() -> Result<StructureLaw> {
    Ok(StructureLaw::SubmergedOrifice(
        SubmergedOrificeParameters::new(
            2.0,
            0.0,
            1.0,
            0.6,
            9.81,
            OrificeApplicability {
                coefficient_provenance: EVIDENCE.into(),
                fully_submerged_applicability: EVIDENCE.into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Gate 2 m wide with sill at 0 m and maximum opening 1 m.
pub fn gate() -> Result<StructureLaw> {
    Ok(StructureLaw::HydrostaticRectangularGate(
        GateParameters::new(
            2.0,
            0.0,
            1.0,
            0.6,
            9.81,
            GateApplicability {
                coefficient_provenance: EVIDENCE.into(),
                hydrostatic_quiescent_pool_assertion: EVIDENCE.into(),
                free_submerged_aperture_applicability: EVIDENCE.into(),
                unobstructed_aperture_assertion: EVIDENCE.into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Pump from A to B.
pub fn pump() -> Result<StructureLaw> {
    Ok(StructureLaw::Pump(PumpParameters::new(
        PumpCurve::new(
            vec![
                PumpCurveKnot {
                    head_m: 0.0,
                    discharge_m3_per_s: 1.0,
                },
                PumpCurveKnot {
                    head_m: 4.0,
                    discharge_m3_per_s: 0.0,
                },
            ],
            EVIDENCE.into(),
        )?,
        -1.0,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: EVIDENCE.into(),
            connection_losses_and_velocity_head_account: EVIDENCE.into(),
            stopped_isolation_assertion: EVIDENCE.into(),
            operating_envelope: EVIDENCE.into(),
        },
    )?))
}

/// Weir, orifice and gate.
pub fn laws() -> Result<[StructureLaw; 3]> {
    Ok([weir()?, orifice()?, gate()?])
}

/// Blockage of `share` on the geometry of `law`. A gate keeps its full opening range.
pub fn blocked(law: &StructureLaw, share: f64) -> Result<StructureCondition> {
    let fraction = BlockageFraction::new(share)?;
    let blockage = match law {
        StructureLaw::SuppressedRectangularWeir(_) => {
            StructureBlockage::weir(fraction, EVIDENCE.into())?
        }
        StructureLaw::SubmergedOrifice(_) => StructureBlockage::orifice(fraction, EVIDENCE.into())?,
        _ => StructureBlockage::gate(fraction, BlockageFraction::NONE, EVIDENCE.into())?,
    };
    Ok(StructureCondition::Blocked(blockage))
}

/// Gate blockage of `width` and `range` shares.
pub fn blocked_gate(width: f64, range: f64) -> Result<StructureCondition> {
    Ok(StructureCondition::Blocked(StructureBlockage::gate(
        BlockageFraction::new(width)?,
        BlockageFraction::new(range)?,
        EVIDENCE.into(),
    )?))
}

pub fn closed() -> StructureCondition {
    StructureCondition::Closed {
        evidence: "Inspection record of the closure".into(),
    }
}

pub fn element(id: &str, law: StructureLaw, condition: StructureCondition) -> StructureElement {
    let direction = if matches!(law, StructureLaw::Pump(_)) {
        DirectionPolicy::AtoBOnly
    } else {
        DirectionPolicy::Bidirectional
    };
    StructureElement {
        id: id.into(),
        endpoint_a: "a".into(),
        endpoint_b: "b".into(),
        datum: DATUM.into(),
        direction,
        law,
        condition,
    }
}

pub fn packet(id: &str, a: HeadSupport, b: HeadSupport) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "a".into(),
            support: a,
        },
        endpoint_b: EndpointHead {
            id: "b".into(),
            support: b,
        },
        datum: DATUM.into(),
        convention: HeadConvention::QuiescentStage,
    }
}

pub fn wet(stage_m: f64) -> HeadSupport {
    HeadSupport::Wet { stage_m }
}

/// Passive operation, or a gate opening of `opening_m`.
pub fn operation(element: &StructureElement, opening_m: f64) -> StructureOperation {
    StructureOperation {
        structure_id: element.id.clone(),
        operating_point: match element.law {
            StructureLaw::HydrostaticRectangularGate(_) => {
                StructureOperatingPoint::GateOpening { opening_m }
            }
            StructureLaw::Pump(_) => StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
            _ => StructureOperatingPoint::Passive,
        },
    }
}

pub fn batch(
    elements: Vec<StructureElement>,
) -> std::result::Result<StructureBatch<Runtime>, StructureError> {
    StructureBatch::<Runtime>::new(crate::runtime::client(), elements)
}

/// Proposal of `elements` at `heads`, each gate at `OPENING`.
pub fn proposal(
    elements: Vec<StructureElement>,
    heads: &[StructureHeads],
) -> Result<StructureProposal<Runtime>> {
    let operations = elements
        .iter()
        .map(|element| operation(element, OPENING))
        .collect::<Vec<_>>();
    Ok(batch(elements)?.propose_with_operations(heads, &operations)?)
}

/// Wet stage pairs over a grid above `floor`, both orders and equal stages included.
pub fn stages(floor: f64, count: usize) -> Vec<[f64; 2]> {
    let value = |index: usize| floor + 0.013 + 2.5 * index as f64 / count as f64;
    let mut pairs = Vec::new();
    for a in 0..count {
        for b in 0..count {
            pairs.push([value(a), value(b)]);
        }
    }
    pairs
}

/// Stage floor of `law`: the weir lanes stay above zero, the orifice lanes above its crown.
pub fn floor(law: &StructureLaw) -> f64 {
    match law {
        StructureLaw::SubmergedOrifice(_) => 1.0,
        _ => 0.0,
    }
}

/// One lane per stage pair, all with `condition`, in pair order.
pub fn lanes(
    law: &StructureLaw,
    condition: &StructureCondition,
    pairs: &[[f64; 2]],
    tag: &str,
) -> (Vec<StructureElement>, Vec<StructureHeads>) {
    pairs
        .iter()
        .enumerate()
        .map(|(index, [a, b])| {
            let id = format!("{tag}-{index}");
            let heads = packet(&id, wet(*a), wet(*b));
            (element(&id, law.clone(), condition.clone()), heads)
        })
        .unzip()
}

pub fn rejected<T>(result: std::result::Result<T, StructureError>) -> Result<String> {
    match result {
        Ok(_) => Err("expected a rejection".into()),
        Err(error) => Ok(error.to_string()),
    }
}
