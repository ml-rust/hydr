//! Certified culvert barrel geometry against frozen independent references.

#[allow(dead_code)]
#[path = "support/runtime.rs"]
mod runtime;

#[path = "culvert_barrels/reference_rows.rs"]
mod reference_rows;

#[path = "culvert_barrels/dense_rows.rs"]
mod dense_rows;

#[path = "culvert_barrels/critical_rows.rs"]
mod critical_rows;

#[path = "culvert_barrels/normal_rows.rs"]
mod normal_rows;

#[path = "culvert_barrels/support.rs"]
mod support;

#[path = "culvert_barrels/geometry.rs"]
mod geometry;

#[path = "culvert_barrels/crown.rs"]
mod crown;

#[path = "culvert_barrels/ranges.rs"]
mod ranges;

#[path = "culvert_barrels/admission.rs"]
mod admission;

#[path = "culvert_barrels/critical.rs"]
mod critical;

#[path = "culvert_barrels/normal.rs"]
mod normal;
