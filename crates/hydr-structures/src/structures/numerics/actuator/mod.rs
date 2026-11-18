//! Original-rate actuator motion, event localization, and atomic acceptance.

mod arithmetic;
pub(crate) mod arrival;
mod arrival_clock;
mod arrival_duration;
mod evaluate;
mod event;
mod forward;
mod fused;
mod initial;
mod motion;
mod step;
mod update;

pub(crate) use evaluate::{evaluate, evaluate_at};
pub(crate) use event::next_event;
pub(crate) use forward::forward;
pub(crate) use initial::initial;
pub(crate) use step::{TensorActuatorLeft, install, left};
pub(crate) use update::admit_update;

#[cfg(test)]
mod count_tests;
