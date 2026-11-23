//! Culvert sediment and face blockage: frozen independent references and open-section geometry.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "culvert_blockage/section_rows.rs"]
mod section_rows;

#[path = "culvert_blockage/sections.rs"]
mod sections;

#[path = "culvert_blockage/floor_lanes.rs"]
mod floor_lanes;

#[allow(dead_code)]
#[path = "culvert_blockage/depth_rows.rs"]
mod depth_rows;

#[path = "culvert_blockage/depths.rs"]
mod depths;

#[path = "culvert_blockage/depth_lanes.rs"]
mod depth_lanes;

#[allow(dead_code)]
#[path = "culvert_blockage/inlet_rows.rs"]
mod inlet_rows;

#[path = "culvert_blockage/inlets.rs"]
mod inlets;

#[allow(dead_code)]
#[path = "culvert_blockage/outlet_rows.rs"]
mod outlet_rows;

#[path = "culvert_blockage/outlets.rs"]
mod outlets;

#[path = "culvert_blockage/outlet_lanes.rs"]
mod outlet_lanes;

#[allow(dead_code)]
#[path = "culvert_blockage/composite_rows.rs"]
mod composite_rows;

#[path = "culvert_blockage/evidence.rs"]
mod evidence;
