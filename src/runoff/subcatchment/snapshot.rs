//! Complete component continuation and atomic checked restoration.

mod package;
mod phase;
mod receipts;
mod restore;
mod source;

pub use package::{SubcatchmentSnapshot, SubcatchmentSnapshotData};
