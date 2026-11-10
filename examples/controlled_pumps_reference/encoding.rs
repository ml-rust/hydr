pub fn object(fields: &[(&str, String)]) -> String {
    format!(
        "{{{}}}",
        fields
            .iter()
            .map(|(key, value)| format!("\"{key}\":{value}"))
            .collect::<Vec<_>>()
            .join(",")
    )
}
pub fn text(value: impl std::fmt::Display) -> String {
    format!("\"{value}\"")
}
pub fn bits(value: f64) -> String {
    text(format!("{:016x}", value.to_bits()))
}
pub fn number(value: usize) -> String {
    value.to_string()
}
pub fn option<T>(value: Option<&T>, encode: impl FnOnce(&T) -> String) -> String {
    value.map_or_else(|| "null".into(), encode)
}
