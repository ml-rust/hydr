//! External-pool inflow ownership, ordering, arithmetic, and continuation checks.

#[path = "coupled_pool_inflows/admission.rs"]
mod admission;
#[path = "coupled_pool_inflows/exchanges.rs"]
mod exchanges;
#[path = "coupled_level_pools/fixture.rs"]
mod fixture;
#[path = "coupled_pool_inflows/inflow_fixture.rs"]
mod inflow_fixture;
#[path = "coupled_level_pools/laws.rs"]
mod laws;
#[path = "coupled_pool_inflows/restoration.rs"]
mod restoration;
#[path = "support/runtime.rs"]
mod runtime;
#[path = "coupled_level_pool_sources/source_fixture.rs"]
mod source_fixture;
