use std::sync::LazyLock;

use super::model::ForwardCase;
use super::{forward_cap_support, forward_conditioning, forward_shapes, forward_translated};

pub(crate) static FORWARD_CASES: LazyLock<Vec<ForwardCase>> = LazyLock::new(|| {
    let mut rows = Vec::with_capacity(79);
    rows.extend_from_slice(&forward_shapes::CASES);
    rows.extend_from_slice(&forward_translated::CASES);
    rows.extend_from_slice(&forward_cap_support::CASES);
    rows.extend_from_slice(&forward_conditioning::CASES);
    rows
});
