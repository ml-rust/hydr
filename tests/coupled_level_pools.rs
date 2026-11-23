//! Coupled-pool conservation, hydraulic accuracy, admission, and restart checks.

#[path = "support/runtime.rs"]
mod runtime;

#[path = "coupled_level_pools/admission.rs"]
mod admission;
#[path = "coupled_level_pools/blockage.rs"]
mod blockage;
#[path = "coupled_level_pools/culverts.rs"]
mod culverts;
#[allow(dead_code)]
#[path = "coupled_level_pools/culverts_fixture.rs"]
mod culverts_fixture;
#[allow(dead_code)]
#[path = "culverts/drain_rows.rs"]
mod drain_rows;
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
