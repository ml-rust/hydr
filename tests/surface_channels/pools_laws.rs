//! Structure laws and operating declarations of the pool fixtures.

use super::{DATUM, Result};
use hydr::coupling::{
    SurfaceChannelActuators, SurfaceChannelControlledGates, SurfaceChannelControlledPumps,
    SurfaceChannelControls, SurfaceChannelNode, SurfaceChannelOperating, SurfaceChannelSignal,
};
use hydr::solver::SolverConfig;
use hydr::structures::{
    ControlCommand, ControlDecisionSchedule, ControlInitialState, ControlLatch, ControlStageSignal,
    CulvertAlignment, CulvertApplicability, CulvertBarrel, CulvertEnd, CulvertEnds, CulvertInlet,
    CulvertParameters, GateActuatorAccuracy, GateActuatorDefinition, GateActuatorInitialState,
    GateActuatorSchedule, GateApplicability, GateMotionLimits, GateOpeningKnot,
    GateOpeningTrajectory, GateParameters, GateTrajectoryAccuracy, GateTrajectoryAssignment,
    OperatingControlDefinition, OrificeApplicability, PumpApplicability, PumpCurve, PumpCurveKnot,
    PumpFreeOutlet, PumpParameters, PumpResponseDefinition, PumpResponseModel, PumpSpeedPolicy,
    SampledControlTiming, StructureLaw, SubmergedOrificeParameters,
    SuppressedRectangularWeirParameters, WeirApplicability,
};

fn gravity() -> f64 {
    SolverConfig::default().gravity
}

/// Circular concrete culvert of `rise` m, `length` m long between `inverts` `[A, B]`.
///
/// Each end has a square-edge headwall: HDS-5 code 1 and entrance loss `0.5`. Manning `n` is
/// `0.013`. `two_way` declares the B end too, else only A-to-B flow has an end.
pub fn culvert_law(
    rise: f64,
    length: f64,
    inverts: [f64; 2],
    two_way: bool,
) -> Result<StructureLaw> {
    barrel_culvert_law(
        CulvertBarrel::circular(rise)?,
        (1, "HDS-5 Table A.1 code 1"),
        length,
        inverts,
        two_way,
    )
}

/// Concrete culvert of `barrel` with HDS-5 inlet `code` at each end, `length` m long between
/// `inverts` `[A, B]`.
///
/// `provenance` names the source of the code's coefficients. The entrance loss is `0.5` and
/// Manning `n` is `0.013`. `two_way` declares the B end too, else only A-to-B flow has an end.
pub fn barrel_culvert_law(
    barrel: CulvertBarrel,
    (code, provenance): (u8, &str),
    length: f64,
    inverts: [f64; 2],
    two_way: bool,
) -> Result<StructureLaw> {
    let end = CulvertEnd::new(CulvertInlet::hds5(code)?, 0.5)?;
    Ok(StructureLaw::Culvert(CulvertParameters::new(
        barrel,
        CulvertAlignment {
            length_m: length,
            invert_a_m: inverts[0],
            invert_b_m: inverts[1],
        },
        0.013,
        1,
        CulvertEnds {
            a: Some(end),
            b: two_way.then_some(end),
        },
        gravity(),
        CulvertApplicability {
            coefficient_provenance: provenance.into(),
            barrel_condition_assertion: "Clean synthetic concrete barrel".into(),
        },
    )?))
}

/// Suppressed weir of `length` m with its crest at `crest`, symmetric under reversal.
pub fn weir_law(crest: f64, length: f64) -> Result<StructureLaw> {
    Ok(StructureLaw::SuppressedRectangularWeir(
        SuppressedRectangularWeirParameters::new(
            length,
            crest,
            1.7,
            WeirApplicability {
                coefficient_provenance: "Declared weir coefficient".into(),
                free_nappe_and_approach_assertion: "Suppressed crest, quiescent approach".into(),
                submergence_correction_assertion: "Villemonte correction adopted".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Fully submerged 0.04 m² orifice between `sill` and `crown`, with `symmetric` reversal.
pub fn orifice_law(sill: f64, crown: f64, symmetric: bool) -> Result<StructureLaw> {
    Ok(StructureLaw::SubmergedOrifice(
        SubmergedOrificeParameters::new(
            0.04,
            sill,
            crown,
            0.6,
            gravity(),
            OrificeApplicability {
                coefficient_provenance: "Declared culvert-approximating coefficient".into(),
                fully_submerged_applicability: "Both sides stand above the crown".into(),
                symmetric_reversal: symmetric,
            },
        )?,
    ))
}

/// 0.5 m wide gate with its sill at `sill`, symmetric under reversal.
pub fn gate_law(sill: f64) -> Result<StructureLaw> {
    Ok(StructureLaw::HydrostaticRectangularGate(
        GateParameters::new(
            0.5,
            sill,
            1.0,
            0.6,
            gravity(),
            GateApplicability {
                coefficient_provenance: "Declared gate coefficient".into(),
                hydrostatic_quiescent_pool_assertion: "Quiescent tanks on both sides".into(),
                free_submerged_aperture_applicability: "One coefficient across regimes".into(),
                unobstructed_aperture_assertion: "Both tanks expose the aperture".into(),
                symmetric_reversal: true,
            },
        )?,
    ))
}

/// Pump on a falling linear curve from 0.2 m³/s at lift -1 m to shutoff at lift 3 m.
///
/// The donor must stay above `suction`. `outlet` adds a free delivery outlet.
pub fn pump_law(suction: f64, outlet: Option<f64>) -> Result<StructureLaw> {
    pump_on(&[(-1.0, 0.2), (3.0, 0.0)], suction, outlet)
}

/// Pump on the piecewise linear curve through `(head, discharge)` knots, its donor above
/// `suction`, with an optional free delivery outlet at `outlet`.
pub fn pump_on(knots: &[(f64, f64)], suction: f64, outlet: Option<f64>) -> Result<StructureLaw> {
    let curve = PumpCurve::new(
        knots
            .iter()
            .map(|&(head_m, discharge_m3_per_s)| PumpCurveKnot {
                head_m,
                discharge_m3_per_s,
            })
            .collect(),
        "Analytical linear curve".into(),
    )?;
    let pump = PumpParameters::new(
        curve,
        suction,
        PumpSpeedPolicy::ReferenceOnly,
        PumpApplicability {
            endpoint_stage_head_assertion: "Quiescent tank stages".into(),
            connection_losses_and_velocity_head_account: "Curve includes connections".into(),
            stopped_isolation_assertion: "Ideal isolation when stopped".into(),
            operating_envelope: "Wet donor within the curve".into(),
        },
    )?;
    Ok(StructureLaw::Pump(match outlet {
        None => pump,
        Some(elevation_m) => pump.with_free_outlet(PumpFreeOutlet {
            elevation_m,
            evidence: "Free discharge at the outlet".into(),
        })?,
    }))
}

/// Controller identity of structure `id`.
pub fn controller(id: &str) -> String {
    format!("relay-{id}")
}

/// The signal of structure `id`'s controller bound to `node`.
pub fn signal(id: &str, node: SurfaceChannelNode) -> SurfaceChannelSignal {
    SurfaceChannelSignal {
        controller_id: controller(id),
        signal_id: format!("signal-{id}"),
        node,
    }
}

/// Two-threshold relay on structure `id`, starting low at `start`, with decisions at `times`.
pub fn relay(
    id: &str,
    [lower, upper]: [f64; 2],
    [low, high]: [ControlCommand; 2],
    (start, times, end): (f64, &[f64], f64),
) -> SurfaceChannelControls {
    SurfaceChannelControls {
        definitions: vec![OperatingControlDefinition {
            controller_id: controller(id),
            structure_id: id.into(),
            signal: ControlStageSignal {
                signal_id: format!("signal-{id}"),
                support_id: format!("support-{id}"),
                datum: DATUM.into(),
            },
            lower_stage_m: lower,
            upper_stage_m: upper,
            high_qualification_delay_s: 0.0,
            low_qualification_delay_s: 0.0,
            low_command: low,
            high_command: high,
            provenance: "Synthetic sampled relay".into(),
        }],
        initial_states: vec![ControlInitialState {
            controller_id: controller(id),
            latch: ControlLatch::Low,
            pending: None,
            antecedent_sample: None,
            last_transition_at_s: None,
        }],
        schedule: ControlDecisionSchedule {
            initial_time_s: start,
            decision_times_s: times.to_vec(),
            end_s: end,
        },
        timing: SampledControlTiming {
            maximum_decision_gap_s: end - start,
            maximum_timer_lateness_s: end - start,
            sampled_applicability: "Synthetic sampled policy".into(),
        },
    }
}

/// Controlled gate `id`: `[low, high]` target openings, actuator `rate` in m/s, starting at
/// opening `low`.
pub fn controlled_gate(
    id: &str,
    thresholds: [f64; 2],
    ([low, high], rate): ([f64; 2], f64),
    schedule: (f64, &[f64], f64),
) -> SurfaceChannelControlledGates {
    let commands = [
        ControlCommand::GateTarget { opening_m: low },
        ControlCommand::GateTarget { opening_m: high },
    ];
    let (start, times, end) = schedule;
    SurfaceChannelControlledGates {
        controls: relay(id, thresholds, commands, schedule),
        actuators: SurfaceChannelActuators {
            definitions: vec![GateActuatorDefinition {
                actuator_id: format!("actuator-{id}"),
                structure_id: id.into(),
                opening_rate_m_per_s: rate,
                closing_rate_m_per_s: rate,
                accuracy: GateActuatorAccuracy {
                    maximum_absolute_opening_error_m: 1e-12,
                    maximum_relative_opening_error: 1e-12,
                    maximum_arrival_window_s: 1e-10,
                },
                rate_provenance: "Synthetic rates".into(),
                ideal_response_applicability: "Ideal rate response".into(),
            }],
            initial_states: vec![GateActuatorInitialState {
                actuator_id: format!("actuator-{id}"),
                opening_m: low,
                target_opening_m: low,
            }],
            schedule: GateActuatorSchedule {
                initial_time_s: start,
                target_times_s: times.to_vec(),
                end_s: end,
            },
        },
    }
}

/// Controlled pump `id` with `[low, high]` commands.
pub fn controlled_pump(
    id: &str,
    thresholds: [f64; 2],
    commands: [ControlCommand; 2],
    schedule: (f64, &[f64], f64),
) -> SurfaceChannelControlledPumps {
    SurfaceChannelControlledPumps {
        controls: relay(id, thresholds, commands, schedule),
        responses: vec![PumpResponseDefinition {
            structure_id: id.into(),
            model: PumpResponseModel::IdealInstantaneous,
            response_provenance: "Synthetic instantaneous operation".into(),
            ideal_response_applicability: "No startup delay".into(),
        }],
    }
}

/// Linear opening path of gate `id` through `(time, opening)` knots.
pub fn trajectory(id: &str, knots: &[(f64, f64)]) -> Result<GateTrajectoryAssignment> {
    Ok(GateTrajectoryAssignment {
        structure_id: id.into(),
        trajectory: GateOpeningTrajectory::new(
            knots
                .iter()
                .map(|&(time_s, opening_m)| GateOpeningKnot { time_s, opening_m })
                .collect(),
            GateMotionLimits {
                maximum_opening_speed_m_per_s: f64::MAX,
                maximum_closing_speed_m_per_s: f64::MAX,
            },
            GateTrajectoryAccuracy {
                maximum_absolute_opening_error_m: 1e-12,
                maximum_relative_opening_error: 1e-12,
            },
            "Synthetic linear trajectory".into(),
        )?,
    })
}

/// Operating children with only `signals` bound.
pub fn operating(signals: Vec<SurfaceChannelSignal>) -> SurfaceChannelOperating {
    SurfaceChannelOperating {
        gates: None,
        pumps: None,
        trajectories: Vec::new(),
        signals,
    }
}
