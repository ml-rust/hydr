use super::{EXTREMES, GEOMETRY, GateReferenceCase, ORDINARY, TRANSITIONS};

pub fn cases() -> impl Iterator<Item = &'static GateReferenceCase> {
    ORDINARY
        .iter()
        .chain(TRANSITIONS)
        .chain(GEOMETRY)
        .chain(EXTREMES)
}
