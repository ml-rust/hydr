use super::model::Case;

pub(crate) const CASES: &[Case] = &[
    super::source_first_knot_forward::CASE,
    super::source_second_knot_reverse::CASE,
    super::three_band_cap_deficit::CASE,
    super::rectangle_cap_deficit::CASE,
    super::first_band_start_signed_offsets::CASE,
    super::middle_band_end_signed_offsets::CASE,
    super::decimal_knot_signed_offsets::CASE,
    super::rectangle_exact_dry::CASE,
    super::rectangle_touching_dry::CASE,
    super::triangle_exact_dry::CASE,
    super::triangle_touching_dry::CASE,
    super::three_band_broad::CASE,
    super::decimal_broad::CASE,
    super::three_band_broad_reverse::CASE,
    super::rectangle_scaled_tiny_positive::CASE,
    super::triangle_scaled_tiny_positive::CASE,
    super::trapezoid_source_retained_offsets::CASE,
];
