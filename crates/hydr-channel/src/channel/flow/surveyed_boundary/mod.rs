//! Certified scalar endpoint traces on surveyed reaches: `Stage`, `Discharge` and `QuiescentReservoir`.

mod bracket;
mod client;
mod contract;
mod depth;
mod error;
#[cfg(test)]
pub(super) mod fixture;
mod interior;
mod point;
mod receipt;
mod request;
mod roots;
mod speed;
mod trace;
