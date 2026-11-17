use super::model::Profile;

pub(crate) const ORIGINAL_RECTANGLE: Profile = Profile {
    coordinates: &[
        [0x0000000000000000, 0x4010000000000000],
        [0x0000000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x4010000000000000],
    ],
    cap_stage: 0x4010000000000000,
};

pub(crate) const LARGE_RECTANGLE: Profile = Profile {
    coordinates: &[
        [0x0000000000000000, 0x54e6dc186ef9f45c],
        [0x0000000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x54e6dc186ef9f45c],
    ],
    cap_stage: 0x54e6dc186ef9f45c,
};
