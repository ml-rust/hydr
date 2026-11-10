//! Sampled relay comparisons and exact anchored-time admission.

mod initial;
mod relay;
mod schedule;
mod selection;
pub(crate) mod timers;

pub(crate) use initial::check_initial;
pub(crate) use relay::decide;
pub(crate) use schedule::check_schedule;
