use super::{
    Case, antecedents, cadence, commands, qualification, sampling, thresholds, timer_extremes,
    timer_fresh, timer_residuals,
};

pub fn cases() -> impl Iterator<Item = &'static Case> {
    thresholds::CASES
        .iter()
        .chain(qualification::CASES)
        .chain(commands::CASES)
        .chain(antecedents::CASES)
        .chain(sampling::CASES)
        .chain(timer_residuals::CASES)
        .chain(timer_extremes::CASES)
        .chain(timer_fresh::CASES)
        .chain(cadence::CASES)
}
