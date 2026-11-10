use super::fixture;
use hydr::structures::{ControlSample, StructureError, StructureInputErrorKind};

pub(super) type SampleChange = (fn(&mut ControlSample), StructureInputErrorKind);

pub(super) fn rejects<T>(
    result: std::result::Result<T, StructureError>,
    kind: StructureInputErrorKind,
    element: Option<usize>,
) -> fixture::Result {
    let error = fixture::rejected(result)?;
    assert_eq!(error.input_kind(), Some(kind), "{error}");
    assert_eq!(error.element(), element, "{error}");
    Ok(())
}
