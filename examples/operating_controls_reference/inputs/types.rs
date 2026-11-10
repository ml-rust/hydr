//! Original control inputs and independently enumerated sampled histories.

use hydr::structures::{ControlCommand, ControlLatch, ControlPending};

#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub observed: f64,
    pub available: f64,
    pub stage: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Timer {
    pub started: f64,
    pub delay: f64,
    pub high: f64,
    pub low: f64,
    pub ready: bool,
    pub allowed: Option<(f64, f64)>,
    pub overflow: bool,
    pub within: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Decision {
    pub sample: Sample,
    pub expected: &'static str,
    pub latch: ControlLatch,
    pub pending: Option<ControlPending>,
    pub transition_count: usize,
    pub transitioned: bool,
    pub qualifies: bool,
    pub accepted_time: f64,
    pub timer: Option<Timer>,
}

#[derive(Clone, Copy, Debug)]
pub struct Case {
    pub family: &'static str,
    pub id: &'static str,
    pub lower: f64,
    pub upper: f64,
    pub high_delay: f64,
    pub low_delay: f64,
    pub maximum_gap: f64,
    pub lateness: f64,
    pub low_command: ControlCommand,
    pub high_command: ControlCommand,
    pub start: f64,
    pub end: f64,
    pub decision_times: &'static [f64],
    pub initial_latch: ControlLatch,
    pub initial_pending: Option<ControlPending>,
    pub antecedent: Option<Sample>,
    pub initial_transition: Option<f64>,
    pub binding: &'static str,
    pub binding_element: Option<usize>,
    pub decisions: &'static [Decision],
}
