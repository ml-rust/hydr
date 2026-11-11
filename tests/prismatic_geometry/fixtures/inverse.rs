use std::sync::LazyLock;

use super::model::InverseCase;
use super::{
    inverse_cap_support, inverse_explicit_areas, inverse_multiband, inverse_polygonal,
    inverse_translated,
};

pub(crate) static INVERSE_CASES: LazyLock<Vec<InverseCase>> = LazyLock::new(|| {
    let mut rows = Vec::with_capacity(86);
    rows.extend_from_slice(&inverse_polygonal::CASES);
    rows.extend_from_slice(&inverse_multiband::CASES);
    rows.extend_from_slice(&inverse_translated::CASES);
    rows.extend_from_slice(&inverse_cap_support::CASES);
    rows.extend_from_slice(&inverse_explicit_areas::CASES);
    rows
});
