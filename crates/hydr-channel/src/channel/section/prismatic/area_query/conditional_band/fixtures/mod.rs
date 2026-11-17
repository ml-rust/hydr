mod loader;
mod metadata;
mod model;
mod parameters;
mod samples;

pub(super) use loader::{FixtureError, load};
pub(super) use model::{
    Bounds, ClipAuthority, FixtureSet, GapAuthority, GapRef, Panel, PanelKind, Path,
    PhysicalEndpoint, RangePolicy, RangeRef, Sample,
};
pub(super) use parameters::{FIELD_ORDER, MAXIMUM_BANDS, PARAMETER_INTERVALS, PARAMETERS};
