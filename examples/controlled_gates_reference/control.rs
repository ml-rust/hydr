use super::proof::*;
use hydr::structures::*;

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
            option(t.allowed_elapsed.as_ref(), |a| {
                object(&[("high", bits(a.high_s)), ("low", bits(a.low_s))])
            }),
        ),
        ("allowed_overflow", t.allowed_elapsed_overflow.to_string()),
        ("within", t.within_lateness.to_string()),
    ])
}
fn transition(t: &ControlTransitionReceipt) -> String {
    object(&[
        ("index", number(t.decision_index)),
        ("time", bits(t.time_s)),
        ("from", text(format!("{:?}", t.from))),
        ("to", text(format!("{:?}", t.to))),
        ("sample", sample(&t.sample)),
        ("timer", timer(&t.timer)),
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
        ("last_sample", option(s.last_sample.as_ref(), sample)),
        ("last_transition_time", float_option(s.last_transition_at_s)),
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
        (
            "target",
            match c.command {
                ControlCommand::GateTarget { opening_m } => bits(opening_m),
                _ => "null".into(),
            },
        ),
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
