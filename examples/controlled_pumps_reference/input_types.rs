use hydr::structures::{ControlCommand, ControlLatch};
#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub time: f64,
    pub available: f64,
    pub stage: f64,
}
#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub id: &'static str,
    pub end: f64,
    pub initial_latch: ControlLatch,
    pub low: ControlCommand,
    pub high: ControlCommand,
    pub high_delay: f64,
    pub low_delay: f64,
    pub gap: f64,
    pub affinity: bool,
    pub adoption: f64,
    pub decisions: &'static [Decision],
    pub queries: &'static [f64],
}
#[derive(Clone, Copy, Debug)]
pub enum Support {
    Wet(f64),
    Unknown,
    MalformedKnownNaN,
}
#[derive(Clone, Copy, Debug)]
pub struct Admission {
    pub id: &'static str,
    pub operation: ControlCommand,
    pub donor: Support,
    pub receiver: Support,
    pub expected: &'static str,
}
