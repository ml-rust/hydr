//! Certified scalar endpoint traces on surveyed reaches: `Stage`, `Discharge` and `QuiescentReservoir`.

mod bracket;
mod capacity;
mod client;
mod contract;
mod depth;
mod error;
#[cfg(test)]
pub(super) mod fixture;
mod interior;
mod momentum;
mod point;
mod receipt;
mod request;
mod roots;
mod speed;
mod trace;

pub(in crate::channel::flow) use capacity::Margin;
pub(in crate::channel::flow) use trace::EXHAUSTED;
