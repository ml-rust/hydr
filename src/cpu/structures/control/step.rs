//! Bounded immediate before/after tensor witnesses without prior-step ownership.

use super::{CpuControlState, CpuControlTimer};
use crate::cpu::CpuTensor;
use crate::structures::ControlSample;

#[derive(Debug)]
pub(crate) struct CpuControlStep {
    pub start: f64,
    pub end: f64,
    pub decision_index: Option<usize>,
    pub before: CpuControlState,
    pub after: CpuControlState,
    pub samples: Vec<ControlSample>,
    pub qualifies: CpuTensor,
    pub timer: CpuControlTimer,
    pub transitioned: CpuTensor,
}
