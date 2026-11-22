//! Surface/channel coordinator definitions, construction checks, common intervals, reach-end
//! ports, pools, structure links, pool ends, operating children, native runoff, and checkpoints.

#[allow(dead_code)]
#[path = "support/analytic.rs"]
mod analytic;
#[allow(dead_code)]
#[path = "channel_pools/reference.rs"]
mod channel_pool_reference;
#[path = "surface_channels/culverts_drain.rs"]
mod culverts_drain;
#[allow(dead_code)]
#[path = "surface_channels/culverts_fixture.rs"]
mod culverts_fixture;
#[path = "surface_channels/culverts_invert.rs"]
mod culverts_invert;
#[path = "surface_channels/culverts_reversal.rs"]
mod culverts_reversal;
#[path = "surface_channels/culverts_roadway.rs"]
mod culverts_roadway;
#[path = "surface_channels/culverts_sweep.rs"]
mod culverts_sweep;
#[path = "surface_channels/definition.rs"]
mod definition;
#[path = "surface_channels/donors.rs"]
mod donors;
#[allow(dead_code)]
#[path = "culverts/drain_rows.rs"]
mod drain_rows;
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
#[path = "surface_channels/pools_donors.rs"]
mod pools_donors;
#[allow(dead_code)]
#[path = "surface_channels/pools_evidence.rs"]
mod pools_evidence;
#[allow(dead_code)]
#[path = "surface_channels/pools_fixture.rs"]
mod pools_fixture;
#[path = "surface_channels/pools_order.rs"]
mod pools_order;
#[path = "surface_channels/pools_orifice_fill.rs"]
mod pools_orifice_fill;
#[path = "surface_channels/pools_reach_end.rs"]
mod pools_reach_end;
#[path = "surface_channels/pools_reach_end_surveyed.rs"]
mod pools_reach_end_surveyed;
#[path = "surface_channels/pools_rest.rs"]
mod pools_rest;
#[path = "surface_channels/pools_restart.rs"]
mod pools_restart;
#[path = "surface_channels/pools_restart_surveyed.rs"]
mod pools_restart_surveyed;
#[path = "surface_channels/pools_weir_drain.rs"]
mod pools_weir_drain;
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
#[path = "surface_channels/ports_restart_surveyed.rs"]
mod ports_restart_surveyed;
#[path = "surface_channels/ports_scenario.rs"]
mod ports_scenario;
#[path = "surface_channels/ports_sections.rs"]
mod ports_sections;
#[path = "surface_channels/ports_seiche.rs"]
mod ports_seiche;
#[path = "surface_channels/ports_step.rs"]
mod ports_step;
#[path = "surface_channels/ports_surveyed.rs"]
mod ports_surveyed;
#[path = "surface_channels/ports_surveyed_dam_break.rs"]
mod ports_surveyed_dam_break;
#[path = "surface_channels/ports_uniform.rs"]
mod ports_uniform;
#[path = "surface_channels/reference.rs"]
mod reference;
#[path = "surface_channels/refinement.rs"]
mod refinement;
#[path = "surface_channels/restoration.rs"]
mod restoration;
#[path = "surface_channels/runoff.rs"]
mod runoff;
#[allow(dead_code)]
#[path = "surface_channels/runoff_fixture.rs"]
mod runoff_fixture;
#[path = "support/runtime.rs"]
mod runtime;
#[path = "surface_channels/scenario.rs"]
mod scenario;
#[path = "surface_channels/structures_gate.rs"]
mod structures_gate;
#[path = "surface_channels/structures_pump.rs"]
mod structures_pump;
#[path = "surface_channels/structures_pump_control.rs"]
mod structures_pump_control;
#[path = "surface_channels/structures_reversal.rs"]
mod structures_reversal;
#[path = "surface_channels/structures_surface_reach.rs"]
mod structures_surface_reach;
#[path = "surface_channels/structures_surface_reach_surveyed.rs"]
mod structures_surface_reach_surveyed;
#[path = "surface_channels/surveyed.rs"]
mod surveyed;
#[allow(dead_code)]
#[path = "surface_channels/surveyed_references.rs"]
mod surveyed_references;
