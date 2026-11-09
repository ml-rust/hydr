//! API-independent original-input reference candidates.

#[derive(Clone, Copy, Debug)]
pub enum ReferenceHead {
    Wet(f64),
    Dry(f64),
    Unknown,
}
#[derive(Clone, Copy, Debug)]
pub enum ReferenceExpectation {
    Success,
    Numerical,
    Unsupported,
}
#[derive(Clone, Copy, Debug)]
pub struct GateReferenceCase {
    pub id: &'static str,
    pub width_m: f64,
    pub sill_m: f64,
    pub maximum_opening_m: f64,
    pub opening_m: f64,
    pub discharge_coefficient: f64,
    pub gravity_m_per_s2: f64,
    pub endpoint_a: ReferenceHead,
    pub endpoint_b: ReferenceHead,
    pub expected: ReferenceExpectation,
    pub reference_discharge: Option<f64>,
    pub relative_tolerance: f64,
    pub subnormal_ulps: u32,
    pub requires_positive: bool,
}
