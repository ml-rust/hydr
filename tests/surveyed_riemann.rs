//! Independent surveyed-profile flux, trace, and runtime checks.

#[path = "surveyed_riemann/accuracy.rs"]
mod accuracy;
#[path = "surveyed_riemann/admission.rs"]
mod admission;
#[path = "../src/channel/flow/surveyed_points/corpus_coverage.rs"]
mod corpus_coverage;
#[path = "surveyed_riemann/data.rs"]
mod data;
#[path = "surveyed_riemann/exact_width.rs"]
mod exact_width;
#[path = "surveyed_riemann/support.rs"]
mod support;
#[path = "surveyed_riemann/wave_domains.rs"]
mod wave_domains;
#[path = "surveyed_riemann/work_budgets.rs"]
mod work_budgets;
