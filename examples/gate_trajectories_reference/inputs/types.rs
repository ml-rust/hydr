//! Original inputs and frozen physical/runtime expectations.

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub family: &'static str,
    pub id: &'static str,
    pub knots: &'static [(f64, f64)],
    pub opening_speed: f64,
    pub closing_speed: f64,
    pub maximum_opening: f64,
    pub absolute: f64,
    pub relative: f64,
    pub binding: &'static str,
    pub static_invalid: bool,
    pub queries: &'static [Query],
}

#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub time: f64,
    pub expected: &'static str,
    pub reference: Option<f64>,
    pub exact_bits: bool,
    pub positive: bool,
    pub closed: bool,
    pub segment: Option<usize>,
    pub knot: Option<usize>,
}
