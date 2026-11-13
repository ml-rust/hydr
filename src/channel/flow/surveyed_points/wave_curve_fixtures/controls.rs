use super::model::{ContractionUnderflow, UncertainSign};

pub(crate) const CONTRACTION_UNDERFLOW: ContractionUnderflow = ContractionUnderflow {
    case_index: 51,
    nonzero_physical_truth: true,
    zero_center_requires_nonzero_radius: true,
};
pub(crate) const UNCERTAIN_SIGN: UncertainSign = UncertainSign {
    case_index: 44,
    delta_center: [0x3af0000000000000, 0x0000000000000000],
    delta_radius: [0x3b00000000000000, 0x0000000000000000],
    actual_shock_offset_unavailable: true,
    overlap_proves_neither_equality_nor_shock: true,
};
pub(crate) const SOURCE_CROSSING: [usize; 2] = [8, 9];
