//! Certified culvert inlet-control headwater against the frozen culvert corpus.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "culverts/coefficient_rows.rs"]
mod coefficient_rows;

#[allow(dead_code)]
#[path = "culverts/inlet_rows.rs"]
mod inlet_rows;

#[allow(dead_code)]
#[path = "culverts/swmm_rows.rs"]
mod swmm_rows;

#[allow(dead_code)]
#[path = "culverts/published_rows.rs"]
mod published_rows;

#[allow(dead_code)]
#[path = "culvert_inlets/lanes.rs"]
mod lanes;

#[path = "culvert_inlets/support.rs"]
mod support;

#[path = "culvert_inlets/corpus.rs"]
mod corpus;

#[path = "culvert_inlets/swmm.rs"]
mod swmm;

#[path = "culvert_inlets/published.rs"]
mod published;

#[path = "culvert_inlets/sweep.rs"]
mod sweep;

#[path = "culvert_inlets/admission.rs"]
mod admission;

#[path = "culvert_inlets/layouts.rs"]
mod layouts;
