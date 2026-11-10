//! Coupled-pool conservation, hydraulic accuracy, admission, and restart checks.

#[path = "support/runtime.rs"]
mod runtime;

#[path = "coupled_level_pools/admission.rs"]
mod admission;
#[path = "coupled_level_pools/exchanges.rs"]
mod exchanges;
#[path = "coupled_level_pools/fixture.rs"]
mod fixture;
#[path = "coupled_level_pools/laws.rs"]
mod laws;
#[path = "coupled_level_pools/refinement.rs"]
mod refinement;
#[path = "coupled_level_pools/restoration.rs"]
mod restoration;
