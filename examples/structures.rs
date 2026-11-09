//! Typed instantaneous laws and held-head volume proposals without storage advancement.

use hydr::structures::{
    DirectionPolicy, EndpointHead, HeadConvention, HeadSupport, HeldHeadInterval,
    OrificeApplicability, StructureBatch, StructureElement, StructureHeads, StructureLaw,
    SubmergedOrificeParameters, SuppressedRectangularWeirParameters, WeirApplicability,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let batch = StructureBatch::new(vec![
        StructureElement {
            id: "opening".into(),
            endpoint_a: "upstream".into(),
            endpoint_b: "downstream".into(),
            datum: "synthetic datum".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::SubmergedOrifice(SubmergedOrificeParameters::new(
                2.0,
                0.0,
                1.0,
                0.6,
                9.81,
                OrificeApplicability {
                    coefficient_provenance: "explicit synthetic Cd=0.6".into(),
                    fully_submerged_applicability: "synthetic quiescent fully submerged opening"
                        .into(),
                    symmetric_reversal: true,
                },
            )?),
        },
        StructureElement {
            id: "crest".into(),
            endpoint_a: "upstream".into(),
            endpoint_b: "downstream".into(),
            datum: "synthetic datum".into(),
            direction: DirectionPolicy::Bidirectional,
            law: StructureLaw::SuppressedRectangularWeir(SuppressedRectangularWeirParameters::new(
                2.0,
                10.0,
                1.7,
                WeirApplicability {
                    coefficient_provenance: "explicit synthetic SI Cw=1.7".into(),
                    free_nappe_and_approach_assertion:
                        "synthetic suppressed sharp crest and declared quiescent approach".into(),
                    submergence_correction_assertion:
                        "adopted Villemonte correction for this synthetic rectangular weir".into(),
                    symmetric_reversal: true,
                },
            )?),
        },
    ])?;
    let proposal = batch.propose(&[heads("opening", 3.0, 2.0), heads("crest", 11.0, 10.5)])?;
    for output in proposal.outputs()? {
        println!(
            "{}: {:?}, {:?}, proposed_Q_m3_per_s={}",
            output.input.structure_id, output.regime, output.status, output.discharge_m3_per_s
        );
    }
    let held = proposal.held_volumes(HeldHeadInterval {
        start: 0.0,
        end: 2.0,
    })?;
    for output in held.outputs()? {
        println!(
            "{}: held_head_volume_proposal_m3={}",
            output.proposal.input.structure_id, output.volume_m3
        );
    }
    let pump=StructureBatch::new(vec![StructureElement{
        id:"pump".into(),endpoint_a:"upstream".into(),endpoint_b:"downstream".into(),
        datum:"synthetic datum".into(),direction:DirectionPolicy::AtoBOnly,
        law:StructureLaw::Pump(hydr::structures::PumpParameters::new(
            hydr::structures::PumpCurve::new(vec![
                hydr::structures::PumpCurveKnot{head_m:0.0,discharge_m3_per_s:4.0},
                hydr::structures::PumpCurveKnot{head_m:4.0,discharge_m3_per_s:2.0},
                hydr::structures::PumpCurveKnot{head_m:8.0,discharge_m3_per_s:0.0},
            ],"analytical SI water curve at declared reference speed 1000 rpm".into())?,
            0.0,hydr::structures::PumpSpeedPolicy::ReferenceOnly,
            hydr::structures::PumpApplicability{
                endpoint_stage_head_assertion:"synthetic quiescent common-datum endpoints".into(),
                connection_losses_and_velocity_head_account:"connection and velocity-head differences are negligible in this synthetic fixture".into(),
                stopped_isolation_assertion:"declared isolation excludes forward siphoning and reverse flow".into(),
                operating_envelope:"known wet donor above stage zero and known wet receiver".into(),
            },
        )?),
    }])?;
    let pump_proposal = pump.propose_with_operations(
        &[heads("pump", 2.0, 6.0)],
        &[hydr::structures::StructureOperation {
            structure_id: "pump".into(),
            operating_point: hydr::structures::StructureOperatingPoint::PumpRunning {
                speed_ratio: 1.0,
            },
        }],
    )?;
    println!("pump_curve_proposal={:?}", pump_proposal.outputs()?);
    Ok(())
}

fn heads(id: &str, a: f64, b: f64) -> StructureHeads {
    StructureHeads {
        structure_id: id.into(),
        endpoint_a: EndpointHead {
            id: "upstream".into(),
            support: HeadSupport::Wet { stage_m: a },
        },
        endpoint_b: EndpointHead {
            id: "downstream".into(),
            support: HeadSupport::Wet { stage_m: b },
        },
        datum: "synthetic datum".into(),
        convention: HeadConvention::QuiescentStage,
    }
}
