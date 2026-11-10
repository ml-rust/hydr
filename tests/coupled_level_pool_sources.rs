//! Connected-pool source order, accounting, forcing admission, and restart checks.

#[path = "coupled_level_pool_sources/admission.rs"]
mod admission;
#[path = "coupled_level_pools/fixture.rs"]
mod fixture;
#[path = "coupled_level_pools/laws.rs"]
mod laws;
#[path = "coupled_level_pool_sources/processes.rs"]
mod processes;
#[path = "coupled_level_pool_sources/restoration.rs"]
mod restoration;
#[path = "support/runtime.rs"]
mod runtime;
#[path = "coupled_level_pool_sources/source_fixture.rs"]
mod source_fixture;
