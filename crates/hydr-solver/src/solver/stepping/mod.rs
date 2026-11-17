//! Accepted-interval trials: source phase, transport completion, and the retry loop.

mod complete;
mod exchange;
mod interval;
mod numerics;
mod prepare;

pub(in crate::solver) use complete::Completion;
pub(in crate::solver) use prepare::{IntervalTrial, Preparation, PreparedInterval};
