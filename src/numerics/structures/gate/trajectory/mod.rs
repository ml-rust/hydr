//! Prescribed-opening motion and point admission using Solvr receipts.

mod accuracy;
mod evaluation;
mod motion;
mod motion_bounds;

pub(crate) use evaluation::evaluate;
pub(crate) use motion::check_motion;
