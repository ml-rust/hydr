//! Original binary64 fixture inputs, without analytical discharge values.

pub(super) enum Expected {
    Running,
    Shutoff,
    Numerical,
    UnsupportedRegime,
}

pub(super) struct Case {
    pub id: &'static str,
    pub donor_stage: f64,
    pub receiving_stage: f64,
    pub speed: f64,
    pub knots: &'static [(f64, f64)],
    pub reference_only: bool,
    pub expected: Expected,
}

#[path = "arithmetic_limits.rs"]
mod arithmetic_limits;
#[path = "operating_points.rs"]
mod operating_points;

pub(super) use arithmetic_limits::CASES as ARITHMETIC_CASES;
pub(super) use operating_points::CASES as OPERATING_CASES;
