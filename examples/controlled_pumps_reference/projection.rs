use super::encoding::*;
use hydr::structures::*;

pub fn operation(p: &StructureOperatingPoint) -> String {
    match p {
        StructureOperatingPoint::PumpStoppedIsolated => {
            object(&[("kind", text("PumpStoppedIsolated"))])
        }
        StructureOperatingPoint::PumpRunning { speed_ratio } => {
            object(&[("kind", text("PumpRunning")), ("speed", bits(*speed_ratio))])
        }
        _ => object(&[("kind", text("UnexpectedNonPumpOperation"))]),
    }
}
pub(super) fn command_value(c: &ControlCommand) -> String {
    match c {
        ControlCommand::PumpStoppedIsolated => {
            operation(&StructureOperatingPoint::PumpStoppedIsolated)
        }
        ControlCommand::PumpRunning { speed_ratio } => {
            operation(&StructureOperatingPoint::PumpRunning {
                speed_ratio: *speed_ratio,
            })
        }
        ControlCommand::GateTarget { opening_m } => {
            object(&[("kind", text("GateTarget")), ("opening", bits(*opening_m))])
        }
    }
}
fn sample(s: &ControlSample) -> String {
    object(&[
        ("controller", text(&s.controller_id)),
        ("signal", text(&s.signal_id)),
        ("support", text(&s.support_id)),
        ("datum", text(&s.datum)),
        ("time", bits(s.observed_at_s)),
        ("available", bits(s.available_at_s)),
        ("stage", bits(s.stage_m)),
    ])
}
fn timer(t: &ControlTimerWitness) -> String {
    object(&[
        ("start", bits(t.started_at_s)),
        ("time", bits(t.evaluated_at_s)),
        ("delay", bits(t.delay_s)),
        ("high", bits(t.elapsed_high_s)),
        ("low", bits(t.elapsed_low_s)),
        ("ready", t.ready.to_string()),
        (
            "allowed",
            option(t.allowed_elapsed.as_ref(), |e| {
                object(&[("high", bits(e.high_s)), ("low", bits(e.low_s))])
            }),
        ),
        ("overflow", t.allowed_elapsed_overflow.to_string()),
        ("within", t.within_lateness.to_string()),
    ])
}
fn transition(r: &ControlTransitionReceipt) -> String {
    object(&[
        ("index", number(r.decision_index)),
        ("time", bits(r.time_s)),
        ("from", text(format!("{:?}", r.from))),
        ("to", text(format!("{:?}", r.to))),
        ("sample", sample(&r.sample)),
        ("timer", timer(&r.timer)),
    ])
}
pub fn state(s: &ControlState) -> String {
    object(&[
        ("controller", text(&s.controller_id)),
        ("latch", text(format!("{:?}", s.latch))),
        (
            "pending",
            option(s.pending.as_ref(), |p| {
                object(&[
                    ("target", text(format!("{:?}", p.target))),
                    ("since", bits(p.since_s)),
                ])
            }),
        ),
        ("sample", option(s.last_sample.as_ref(), sample)),
        (
            "transition_time",
            s.last_transition_at_s.map_or_else(|| "null".into(), bits),
        ),
        ("transitions", number(s.transition_count)),
        (
            "last_transition",
            option(s.last_transition.as_ref(), transition),
        ),
    ])
}
pub fn command(c: &ControlCommandOutput) -> String {
    object(&[
        ("element", number(c.element)),
        ("controller", text(&c.controller_id)),
        ("structure", text(&c.structure_id)),
        ("time", bits(c.time_s)),
        ("latch", text(format!("{:?}", c.latch))),
        ("operation", command_value(&c.command)),
    ])
}
fn receipt(r: &ControlDecisionReceipt) -> String {
    object(&[
        ("element", number(r.element)),
        ("controller", text(&r.controller_id)),
        ("before", state(&r.before)),
        ("sample", sample(&r.sample)),
        ("qualifies", r.qualifies.to_string()),
        ("timer", option(r.timer.as_ref(), timer)),
        ("transitioned", r.transitioned.to_string()),
        ("after", state(&r.after)),
    ])
}
pub fn step(s: &ControlStepDiagnostics, lane: usize) -> String {
    object(&[
        ("start", bits(s.start_s)),
        ("end", bits(s.end_s)),
        (
            "index",
            s.decision_index.map_or_else(|| "null".into(), number),
        ),
        ("left", option(s.left_commands.get(lane), command)),
        ("right", option(s.right_commands.get(lane), command)),
        ("receipt", option(s.decisions.get(lane), receipt)),
    ])
}
pub fn response(o: &ControlledPumpOutput, d: &PumpResponseDefinition) -> String {
    object(&[
        ("element", number(o.element)),
        ("controller", text(&o.controller_id)),
        ("structure", text(&o.structure_id)),
        ("time", bits(o.time_s)),
        ("steps", number(o.accepted_steps)),
        ("model", text(format!("{:?}", o.model))),
        ("latch", text(format!("{:?}", o.latch))),
        ("operation", operation(&o.operating_point)),
        ("provenance", text(&d.response_provenance)),
        ("applicability", text(&d.ideal_response_applicability)),
    ])
}
fn endpoint(e: &EndpointHead) -> String {
    object(&[
        ("id", text(&e.id)),
        (
            "support",
            match &e.support {
                HeadSupport::Unknown => object(&[("kind", text("Unknown"))]),
                HeadSupport::Wet { stage_m } => {
                    object(&[("kind", text("Wet")), ("stage", bits(*stage_m))])
                }
                HeadSupport::Dry { bed_elevation_m } => {
                    object(&[("kind", text("Dry")), ("bed", bits(*bed_elevation_m))])
                }
            },
        ),
    ])
}
pub fn heads(h: &StructureHeads) -> String {
    object(&[
        ("structure", text(&h.structure_id)),
        ("a", endpoint(&h.endpoint_a)),
        ("b", endpoint(&h.endpoint_b)),
        ("datum", text(&h.datum)),
        ("convention", text(format!("{:?}", h.convention))),
    ])
}
pub fn source(s: &OperatingSourceReceipt) -> String {
    object(&[
        ("source", text(&s.source_id)),
        ("element", number(s.element)),
        ("structure", text(&s.structure_id)),
        ("time", bits(s.time_s)),
        (
            "detail",
            match &s.detail {
                OperatingSourceDetail::ControlledPump {
                    controller_id,
                    accepted_steps,
                    accepted_time_s,
                    latch,
                    response_model,
                    operating_point,
                } => object(&[
                    ("kind", text("ControlledPump")),
                    ("controller", text(controller_id)),
                    ("steps", number(*accepted_steps)),
                    ("accepted_time", bits(*accepted_time_s)),
                    ("latch", text(format!("{latch:?}"))),
                    ("model", text(format!("{response_model:?}"))),
                    ("operation", operation(operating_point)),
                ]),
                _ => object(&[("kind", text("UnexpectedSource"))]),
            },
        ),
    ])
}
fn operands(p: &PumpHydraulicOperands) -> String {
    object(&[
        ("speed", bits(p.speed_ratio())),
        ("head_high", bits(p.receiving_minus_donor_head_m())),
        ("head_low", bits(p.receiving_minus_donor_head_low_m())),
        ("query", bits(p.reference_head_m())),
        ("error", bits(p.normalization_error_bound_m())),
        ("lower", bits(p.reference_head_lower_m())),
        ("upper", bits(p.reference_head_upper_m())),
        ("reference_q", bits(p.reference_discharge_m3_per_s())),
        ("division_exact", p.division_exact().to_string()),
    ])
}
pub fn hydraulic(h: &StructureOutput) -> String {
    object(&[
        ("element", number(h.element)),
        ("operation", operation(&h.operation)),
        ("input", heads(&h.input)),
        ("law", text(format!("{:?}", h.law))),
        ("direction", text(format!("{:?}", h.direction_policy))),
        ("regime", text(format!("{:?}", h.regime))),
        ("status", text(format!("{:?}", h.status))),
        ("q", bits(h.discharge_m3_per_s)),
        (
            "operands",
            match &h.hydraulic_operands {
                Some(StructureHydraulicOperands::Pump(p)) => operands(p),
                None => "null".into(),
                _ => object(&[("unexpected", text("nonpump operands"))]),
            },
        ),
    ])
}
