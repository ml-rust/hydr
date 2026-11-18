//! Surface/channel coordinator definitions, construction checks, common intervals, reach-end
//! ports, pools, structure links, pool ends, operating children, and checkpoints.

#[allow(dead_code)]
#[path = "support/analytic.rs"]
mod analytic;
#[path = "surface_channels/definition.rs"]
mod definition;
#[path = "surface_channels/donors.rs"]
mod donors;
#[path = "surface_channels/fixture.rs"]
mod fixture;
#[path = "surface_channels/frozen.rs"]
mod frozen;
#[path = "surface_channels/identity.rs"]
mod identity;
#[path = "support/impervious.rs"]
mod impervious;
#[path = "surface_channels/levelling.rs"]
mod levelling;
#[path = "surface_channels/lifecycle.rs"]
mod lifecycle;
#[path = "surface_channels/physics.rs"]
mod physics;
#[path = "surface_channels/pools_definition.rs"]
mod pools_definition;
#[allow(dead_code)]
#[path = "surface_channels/pools_fixture.rs"]
mod pools_fixture;
#[path = "surface_channels/pools_order.rs"]
mod pools_order;
#[path = "surface_channels/pools_restart.rs"]
mod pools_restart;
#[path = "surface_channels/ports_dam_break.rs"]
mod ports_dam_break;
#[path = "surface_channels/ports_definition.rs"]
mod ports_definition;
#[path = "surface_channels/ports_donors.rs"]
mod ports_donors;
#[path = "surface_channels/ports_fixture.rs"]
mod ports_fixture;
#[path = "surface_channels/ports_limits.rs"]
mod ports_limits;
#[path = "surface_channels/ports_line.rs"]
mod ports_line;
#[path = "surface_channels/ports_order.rs"]
mod ports_order;
#[path = "surface_channels/ports_rest.rs"]
mod ports_rest;
#[path = "surface_channels/ports_restart.rs"]
mod ports_restart;
#[path = "surface_channels/ports_scenario.rs"]
mod ports_scenario;
#[path = "surface_channels/ports_seiche.rs"]
mod ports_seiche;
#[path = "surface_channels/ports_step.rs"]
mod ports_step;
#[path = "surface_channels/ports_uniform.rs"]
mod ports_uniform;
#[path = "surface_channels/reference.rs"]
mod reference;
#[path = "surface_channels/refinement.rs"]
mod refinement;
#[path = "surface_channels/restoration.rs"]
mod restoration;
#[path = "support/runtime.rs"]
mod runtime;
#[path = "surface_channels/scenario.rs"]
mod scenario;
#[path = "surface_channels/surveyed.rs"]
mod surveyed;
