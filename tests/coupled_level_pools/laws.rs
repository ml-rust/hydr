use super::fixture::{DATUM, Result};
use hydr::structures::{
    DirectionPolicy, PumpApplicability, PumpCurve, PumpCurveKnot, PumpParameters, PumpSpeedPolicy,
    StructureElement, StructureLaw, StructureOperatingPoint, StructureOperation,
    SuppressedRectangularWeirParameters, WeirApplicability,
};

pub fn weir(id: &str, a: &str, b: &str, crest: f64, coefficient: f64) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: a.into(),
        endpoint_b: b.into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::Bidirectional,
        law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
            1.0,
            crest,
            coefficient,
            WeirApplicability {
                coefficient_provenance: "Explicit synthetic free-weir coefficient".into(),
                free_nappe_and_approach_assertion:
                    "Suppressed sharp crest with quiescent synthetic approaches".into(),
                submergence_correction_assertion:
                    "Declared existing Villemonte correction where applicable".into(),
                symmetric_reversal: true,
            },
        )?),
    })
}

pub fn pump(id: &str, a: &str, b: &str, knots: &[(f64, f64)]) -> Result<StructureElement> {
    pump_with_suction(id, a, b, knots, 0.0)
}

pub fn pump_with_suction(
    id: &str,
    a: &str,
    b: &str,
    knots: &[(f64, f64)],
    minimum_suction_stage_m: f64,
) -> Result<StructureElement> {
    Ok(StructureElement {
        id: id.into(),
        endpoint_a: a.into(),
        endpoint_b: b.into(),
        datum: DATUM.into(),
        direction: DirectionPolicy::AtoBOnly,
        law: StructureLaw::Pump(PumpParameters::new(
            PumpCurve::new(
                knots
                    .iter()
                    .map(|&(head_m, discharge_m3_per_s)| PumpCurveKnot {
                        head_m,
                        discharge_m3_per_s,
                    })
                    .collect(),
                "Explicit synthetic SI reference-speed curve".into(),
            )?,
            minimum_suction_stage_m,
            PumpSpeedPolicy::ReferenceOnly,
            PumpApplicability {
                endpoint_stage_head_assertion: "Quiescent common-datum synthetic endpoint stages"
                    .into(),
                connection_losses_and_velocity_head_account:
                    "Effective reference curve includes declared connection losses".into(),
                stopped_isolation_assertion:
                    "Explicit isolation prevents siphoning and reverse flow when stopped".into(),
                operating_envelope:
                    "Synthetic numerical fixture with positive wet suction and receiving support"
                        .into(),
            },
        )?),
    })
}

pub fn passive(id: &str) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point: StructureOperatingPoint::Passive,
    }
}

pub fn running(id: &str) -> StructureOperation {
    StructureOperation {
        structure_id: id.into(),
        operating_point: StructureOperatingPoint::PumpRunning { speed_ratio: 1.0 },
    }
}
