//! Safe Rust interface over the pinned EPA SWMM v5.2.4 drainage worker.
//!
//! Each [`drainage::SwmmDrainage`] instance runs an isolated worker process so SWMM's
//! process-global C state never aliases across instances and the workspace keeps
//! `#![forbid(unsafe_code)]`.

pub mod drainage;
