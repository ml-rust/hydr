pub(crate) const PARAMETERS: &[u64] = &[
    0x0000000000000000,
    0x3fd0000000000000,
    0x3fe0000000000000,
    0x3fe8000000000000,
    0x3ff0000000000000,
];
pub(crate) const PARAMETER_INTERVALS: &[[u64; 2]] = &[
    [0x0000000000000000, 0x3ff0000000000000],
    [0x0000000000000000, 0x3fd0000000000000],
    [0x3fd0000000000000, 0x3fe0000000000000],
    [0x3fe0000000000000, 0x3fe8000000000000],
    [0x3fe8000000000000, 0x3ff0000000000000],
    [0x3fd0000000000000, 0x3fe8000000000000],
];
pub(crate) const MAXIMUM_BANDS: usize = 3;
pub(crate) const FIELD_ORDER: [&str; 10] = [
    "area",
    "depth",
    "width",
    "pressure_integral",
    "hydraulic_depth",
    "sound",
    "lower_area_distance",
    "upper_area_distance",
    "lower_depth_distance",
    "upper_depth_distance",
];
