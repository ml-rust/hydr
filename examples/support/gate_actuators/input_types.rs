//! Original input and policy bits only. Derived physical truth remains in frozen external JSON.
use hydr::structures::GateActuatorAccuracy;
#[derive(Clone, Copy, Debug)]
pub struct Update {
    pub time: f64,
    pub target: f64,
    pub available: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Query {
    pub time: f64,
    pub expected: &'static str,
}
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub family: &'static str,
    pub id: &'static str,
    pub policy: &'static str,
    pub start: f64,
    pub end: f64,
    pub initial: f64,
    pub target: f64,
    pub opening_rate: f64,
    pub closing_rate: f64,
    pub maximum: f64,
    pub accuracy: GateActuatorAccuracy,
    pub updates: &'static [Update],
    pub queries: &'static [Query],
    pub binding_expected: &'static str,
}
