use super::{
    Case, films, motion_certificates, motion_limits, nonbinary, profiles, supplements, support,
    time_scales,
};

pub fn cases() -> impl Iterator<Item = &'static Case> {
    profiles::CASES
        .iter()
        .chain(nonbinary::CASES)
        .chain(motion_limits::CASES)
        .chain(time_scales::CASES)
        .chain(films::CASES)
        .chain(support::CASES)
        .chain(supplements::CASES)
        .chain(motion_certificates::CASES)
}
