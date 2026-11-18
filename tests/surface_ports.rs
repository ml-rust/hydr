//! Coordinator surface port faces through the hidden surface API.

#[path = "surface_ports/accounting.rs"]
mod accounting;
#[path = "surface_ports/admission.rs"]
mod admission;
#[path = "surface_ports/fixture.rs"]
mod fixture;
#[path = "surface_ports/flux.rs"]
mod flux;
#[path = "surface_ports/limits.rs"]
mod limits;
#[path = "surface_ports/rest.rs"]
mod rest;
#[path = "support/runtime.rs"]
mod runtime;
