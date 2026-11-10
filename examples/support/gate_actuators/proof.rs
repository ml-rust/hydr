use hydr::structures::*;

pub fn object(fields: &[(&str, String)]) -> String {
    let mut output = String::from("{");
    for (index, (key, value)) in fields.iter().enumerate() {
        if index != 0 {
            output.push(',');
        }
        output.push('"');
        output.push_str(key);
        output.push_str("\":");
        output.push_str(value);
    }
    output.push('}');
    output
}
pub fn text(value: impl std::fmt::Display) -> String {
    format!("\"{value}\"")
}
pub fn bits(value: f64) -> String {
    format!("\"{:016x}\"", value.to_bits())
}
pub fn option<T>(value: Option<&T>, project: impl FnOnce(&T) -> String) -> String {
    value.map_or_else(|| "null".into(), project)
}
pub fn number(value: usize) -> String {
    value.to_string()
}
pub fn float_option(value: Option<f64>) -> String {
    value.map_or_else(|| "null".into(), bits)
}
pub fn position(p: &GateActuatorPosition) -> String {
    object(&[
        ("center", bits(p.opening_m)),
        ("lower", bits(p.lower_opening_m)),
        ("upper", bits(p.upper_opening_m)),
        ("error", bits(p.absolute_error_m)),
    ])
}
pub fn anchor(a: &GateActuatorAnchor) -> String {
    object(&[
        ("time", bits(a.time_s)),
        ("position", position(&a.position)),
    ])
}
pub fn expansion(t: &GateActuatorTimeExpansion) -> String {
    object(&[("high", bits(t.high_s)), ("low", bits(t.low_s))])
}
pub fn arrival(a: &GateActuatorArrivalCertificate) -> String {
    object(&[
        ("anchor", anchor(&a.anchor)),
        ("target", bits(a.target_opening_m)),
        ("lower", bits(a.lower_s)),
        ("upper", bits(a.upper_s)),
        ("width", expansion(&a.width_s)),
    ])
}
fn binary(w: &GateActuatorBinaryWitness) -> String {
    object(&[
        ("product", bits(w.product_m)),
        ("high", bits(w.sum_high_m)),
        ("low", bits(w.sum_low_m)),
    ])
}
fn stage(s: &GateActuatorFmaStageReceipt) -> String {
    object(&[
        ("rate", bits(s.signed_rate_m_per_s)),
        ("time", bits(s.time_component_s)),
        ("addend", bits(s.addend_m)),
        ("result", bits(s.result_m)),
        ("error", bits(s.rounding_error_m)),
        ("zero_grid", s.zero_grid_certificate.to_string()),
        ("binary", option(s.exact_binary_witness.as_ref(), binary)),
    ])
}
pub fn motion(m: &GateActuatorMotionReceipt) -> String {
    object(&[
        ("direction", text(format_args!("{:?}", m.direction))),
        ("rate", bits(m.rate_m_per_s)),
        ("primary", stage(&m.primary)),
        ("elapsed_low", option(m.elapsed_low.as_ref(), stage)),
        ("raw_error", bits(m.raw_error_m)),
        ("raw_lower", float_option(m.raw_lower_m)),
        ("raw_upper", float_option(m.raw_upper_m)),
    ])
}
fn endpoint(e: &GateActuatorEndpointReceipt) -> String {
    object(&[
        ("anchor", bits(e.anchor_opening_m)),
        ("motion", option(e.motion.as_ref(), motion)),
        ("lower", bits(e.lower_opening_m)),
        ("upper", bits(e.upper_opening_m)),
    ])
}
pub fn forward(f: &GateActuatorForwardReceipt) -> String {
    object(&[
        ("anchor", anchor(&f.anchor)),
        ("query", bits(f.query_time_s)),
        ("target", bits(f.target_opening_m)),
        ("elapsed", expansion(&f.elapsed_s)),
        ("lower_endpoint", endpoint(&f.lower_endpoint)),
        ("upper_endpoint", endpoint(&f.upper_endpoint)),
        ("center_motion", option(f.center_motion.as_ref(), motion)),
        ("position", position(&f.position)),
    ])
}
pub fn completion(c: &GateActuatorCompletionReceipt) -> String {
    object(&[
        ("episode", number(c.episode_index)),
        ("generation", number(c.target_generation)),
        ("confirmed", bits(c.confirmed_at_s)),
        ("original_arrival", arrival(&c.original_arrival)),
        ("effective_upper", bits(c.effective_upper_s)),
        ("effective_width", expansion(&c.effective_width_s)),
        ("forward", option(c.forward.as_ref(), forward)),
    ])
}
pub fn update(u: &GateTargetUpdate) -> String {
    object(&[
        ("actuator", text(&u.actuator_id)),
        ("structure", text(&u.structure_id)),
        ("target", bits(u.target_opening_m)),
        ("issued", bits(u.issued_at_s)),
        ("available", bits(u.available_at_s)),
    ])
}
pub fn event(e: &GateActuatorEvent) -> String {
    object(&[
        ("time", bits(e.time_s)),
        (
            "target_index",
            e.target_index.map_or_else(|| "null".into(), number),
        ),
        ("lower", e.arrival_lower_boundary.to_string()),
        ("upper", e.arrival_upper_boundary.to_string()),
        ("terminal", e.terminal.to_string()),
    ])
}
