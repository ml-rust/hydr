//! HDS-5 culvert structure law: inversion, direction, zero flow, sensitivity and layouts.

#[allow(dead_code)]
#[path = "culvert_laws/support.rs"]
mod support;

#[allow(dead_code)]
#[path = "culverts/inlet_rows.rs"]
mod inlet_rows;

#[allow(dead_code)]
#[path = "culverts/outlet_rows.rs"]
mod outlet_rows;

#[path = "culvert_laws/corpus.rs"]
mod corpus;

#[path = "culvert_laws/round_trip.rs"]
mod round_trip;

#[path = "culvert_laws/heads.rs"]
mod heads;

#[allow(dead_code)]
#[path = "culvert_laws/lanes.rs"]
mod lanes;

#[path = "culvert_laws/layouts.rs"]
mod layouts;

#[path = "culvert_laws/sweep.rs"]
mod sweep;

#[path = "culvert_laws/capacity.rs"]
mod capacity;
