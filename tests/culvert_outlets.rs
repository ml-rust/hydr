//! Certified culvert outlet-control headwater against the frozen culvert corpus.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[allow(dead_code)]
#[path = "culverts/outlet_rows.rs"]
mod outlet_rows;

#[allow(dead_code)]
#[path = "culverts/published_rows.rs"]
mod published_rows;

#[allow(dead_code)]
#[path = "culvert_outlets/lanes.rs"]
mod lanes;

#[allow(dead_code)]
#[path = "culvert_outlets/support.rs"]
mod support;

#[path = "culvert_outlets/corpus.rs"]
mod corpus;

#[path = "culvert_outlets/sweep.rs"]
mod sweep;

#[path = "culvert_outlets/published.rs"]
mod published;

#[path = "culvert_outlets/layouts.rs"]
mod layouts;

#[path = "culvert_outlets/admission.rs"]
mod admission;
