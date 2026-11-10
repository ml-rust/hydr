use super::proof::*;
use hydr::structures::*;

fn change(c: &GateActuatorTargetChangeReceipt) -> String {
    object(&[
        ("index", number(c.target_index)),
        ("time", bits(c.time_s)),
        ("previous_target", bits(c.previous_target_opening_m)),
        ("previous_anchor", anchor(&c.previous_anchor)),
        (
            "previous_completion",
            option(c.previous_completion.as_ref(), completion),
        ),
        ("update", update(&c.update)),
        ("left_position", position(&c.left_position)),
        ("new_anchor", anchor(&c.new_anchor)),
        ("generation", number(c.target_generation)),
        (
            "new_episode",
            c.new_episode_index.map_or_else(|| "null".into(), number),
        ),
    ])
}
pub fn state(s: &GateActuatorState) -> String {
    object(&[
        ("actuator", text(&s.actuator_id)),
        ("anchor", anchor(&s.anchor)),
        ("target", bits(s.target_opening_m)),
        ("position", position(&s.position)),
        ("phase", text(format_args!("{:?}", s.phase))),
        ("arrival", option(s.arrival.as_ref(), arrival)),
        ("target_changes", number(s.target_change_count)),
        ("episodes", number(s.episode_count)),
        ("arrivals", number(s.arrival_count)),
        (
            "active_episode",
            s.active_episode_index.map_or_else(|| "null".into(), number),
        ),
        ("latest_update", option(s.latest_update.as_ref(), update)),
        ("last_change", option(s.last_target_change.as_ref(), change)),
        (
            "last_completion",
            option(s.last_completion.as_ref(), completion),
        ),
    ])
}
pub fn output(o: &GateActuatorOutput) -> String {
    object(&[
        ("element", number(o.element)),
        ("actuator", text(&o.actuator_id)),
        ("structure", text(&o.structure_id)),
        ("time", bits(o.time_s)),
        ("target", bits(o.target_opening_m)),
        ("position", position(&o.position)),
        ("phase", text(format_args!("{:?}", o.phase))),
        ("support", text(format_args!("{:?}", o.support))),
        ("arrival_lower", float_option(o.arrival_lower_s)),
        ("arrival_upper", float_option(o.arrival_upper_s)),
    ])
}
pub fn receipt(r: &GateActuatorStepReceipt) -> String {
    object(&[
        ("element", number(r.element)),
        ("actuator", text(&r.actuator_id)),
        ("before", state(&r.before)),
        ("left_position", position(&r.left_position)),
        ("left_forward", option(r.left_forward.as_ref(), forward)),
        ("update", option(r.update.as_ref(), update)),
        ("completed", r.completed_episode.to_string()),
        ("changed", r.target_changed.to_string()),
        ("after", state(&r.after)),
    ])
}
