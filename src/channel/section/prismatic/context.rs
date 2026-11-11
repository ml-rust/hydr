use super::super::{CrossSectionError, context::Context as SectionContext};
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;
use solvr::common::arithmetic::{ArithmeticError, Interval};
use std::sync::Arc;

#[derive(Debug)]
pub(super) struct Context<R: Runtime<DType = DType>> {
    pub source: Arc<SectionContext<R>>,
    pub maximum_bands: usize,
    pub valid: Tensor<R>,
    pub pointed: Tensor<R>,
    pub start_depth: Interval<R>,
    pub end_depth: Interval<R>,
    pub thickness: Interval<R>,
    pub start_area: Interval<R>,
    pub end_area: Interval<R>,
    pub area_increment: Interval<R>,
    pub start_width: Interval<R>,
    pub slope: Interval<R>,
    pub start_integral: Interval<R>,
    pub cap_depth: Interval<R>,
    pub cap_area: Interval<R>,
    pub cap_width: Interval<R>,
    pub cap_integral: Interval<R>,
}

pub(super) fn arithmetic_error(error: ArithmeticError) -> CrossSectionError {
    match error {
        ArithmeticError::Range { .. } | ArithmeticError::InvalidBounds => {
            CrossSectionError::numerical(format!("prismatic geometry: {error}"))
        }
        other => CrossSectionError::Runtime {
            section: None,
            context: format!("prismatic geometry: {other}"),
        },
    }
}
