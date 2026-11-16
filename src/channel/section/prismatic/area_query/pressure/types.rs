use numr::{dtype::DType, runtime::Runtime};
use solvr::common::arithmetic::{CertifiedExpansion, Interval};

pub(crate) struct PressureEndpoint<R: Runtime<DType = DType>> {
    pub(crate) delta_area: CertifiedExpansion<R>,
    /// Carries physical map uncertainty and arithmetic errors. Keep selected-point loss accounting separate.
    pub(crate) retained_increment: CertifiedExpansion<R>,
    pub(crate) physical_increment: Interval<R>,
    pub(crate) secant: Interval<R>,
}

/// Physical endpoint extrema enclose the complete supported query range.
#[cfg(test)]
pub(crate) struct PressureRange<R: Runtime<DType = DType>> {
    pub(crate) lower: PressureEndpoint<R>,
    pub(crate) upper: PressureEndpoint<R>,
}
