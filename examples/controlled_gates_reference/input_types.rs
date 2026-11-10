use hydr::structures::GateActuatorAccuracy;
#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub time: f64,
    pub available: f64,
    pub stage: f64,
    pub admitted: bool,
}
#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub time: f64,
    pub expected: &'static str,
}
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub id: &'static str,
    pub start: f64,
    pub end: f64,
    pub initial: f64,
    pub target: f64,
    pub opening_rate: f64,
    pub closing_rate: f64,
    pub low: f64,
    pub high: f64,
    pub lower_stage: f64,
    pub upper_stage: f64,
    pub high_delay: f64,
    pub low_delay: f64,
    pub gap: f64,
    pub lateness: f64,
    pub accuracy: GateActuatorAccuracy,
    pub decisions: &'static [Decision],
    pub queries: &'static [Query],
}
