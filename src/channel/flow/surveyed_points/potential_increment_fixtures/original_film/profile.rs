use super::super::model::Profile;

pub(crate) const PROFILE: Profile = Profile {
    id: "rectangle",
    coordinates: &[
        [0x0000000000000000, 0x4010000000000000],
        [0x0000000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x0000000000000000],
        [0x3ff0000000000000, 0x4010000000000000],
    ],
    cap_stage: 0x4010000000000000,
};
