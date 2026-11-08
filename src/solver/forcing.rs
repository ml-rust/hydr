use super::{BoundaryConditions, EngineError, FluxPort};
use crate::model::Domain;
use std::collections::BTreeSet;

/// Piecewise-constant spatial forcing on the half-open interval `[start, end)`.
#[derive(Clone, Debug, PartialEq)]
pub struct ForcingInterval {
    /// Elapsed start time in seconds.
    pub start: f64,
    /// Elapsed end time in seconds.
    pub end: f64,
    /// Cell-supported vertical rain in m/s, with zero horizontal momentum.
    pub rain: Vec<f64>,
    /// Prescribed land-loss demand in m/s. Removes proportional donor momentum.
    ///
    /// This is no infiltration model. Unmet demand rejects a trial without capping demand.
    pub withdrawal: Vec<f64>,
    /// Source-complete boundary rates, including pressure and declared speed bounds.
    pub ports: Vec<FluxPort>,
}

/// Checked forcing coverage with no gaps, implicit rain, or provider callbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct ForcingSchedule {
    intervals: Vec<ForcingInterval>,
}

impl ForcingSchedule {
    /// Checks spatial arrays, contiguous interval support, and complete port assignments.
    ///
    /// # Errors
    /// Rejects gaps, overlap, nonfinite inputs, negative rates, or missing/duplicate ports.
    pub fn new(
        domain: &Domain,
        boundaries: &BoundaryConditions,
        intervals: Vec<ForcingInterval>,
    ) -> Result<Self, EngineError> {
        if intervals.is_empty() {
            return Err(EngineError::rejected(
                "forcing",
                "no intervals: supply explicit coverage",
            ));
        }
        let declared: BTreeSet<_> = boundaries.ports().iter().copied().collect();
        for (index, interval) in intervals.iter().enumerate() {
            if !interval.start.is_finite()
                || !interval.end.is_finite()
                || interval.start < 0.0
                || interval.end <= interval.start
                || (index > 0 && intervals[index - 1].end != interval.start)
            {
                return Err(EngineError::rejected(
                    "forcing",
                    format!(
                        "interval {index} [{}, {}): supply finite contiguous increasing support",
                        interval.start, interval.end
                    ),
                ));
            }
            for (name, rates) in [
                ("rain", &interval.rain),
                ("withdrawal", &interval.withdrawal),
            ] {
                if rates.len() != domain.grid().cell_count() {
                    return Err(EngineError::rejected(
                        "forcing",
                        format!(
                            "interval {index} {name} shape {}: require {} cells",
                            rates.len(),
                            domain.grid().cell_count()
                        ),
                    ));
                }
                for (cell, rate) in rates.iter().enumerate() {
                    if !rate.is_finite() || *rate < 0.0 || (!domain.active()[cell] && *rate != 0.0)
                    {
                        return Err(EngineError::rejected(
                            "forcing",
                            format!(
                                "interval {index} {name}[{cell}]={rate}: require finite nonnegative active-cell rates and zero inactive rates"
                            ),
                        ));
                    }
                }
            }
            if interval.ports.len() != boundaries.ports().len() {
                return Err(EngineError::rejected(
                    "forcing",
                    format!("interval {index}: supply exactly one rate for each declared port"),
                ));
            }
            let mut seen = BTreeSet::new();
            for (port_index, port) in interval.ports.iter().enumerate() {
                if !declared.contains(&port.face)
                    || !seen.insert(port.face)
                    || !port.water_rate.is_finite()
                    || port.momentum_rate.iter().any(|value| !value.is_finite())
                    || !port.signal_speed.is_finite()
                    || port.signal_speed <= 0.0
                    || port.provenance.trim().is_empty()
                {
                    return Err(EngineError::rejected(
                        "forcing",
                        format!(
                            "interval {index} port {port_index}={port:?}: require unique declared support, finite complete flux, positive speed bound, and regime/bound provenance"
                        ),
                    ));
                }
            }
        }
        Ok(Self { intervals })
    }

    /// Returns immutable forcing intervals.
    pub fn intervals(&self) -> &[ForcingInterval] {
        &self.intervals
    }

    pub(super) fn at(&self, time: f64) -> Option<&ForcingInterval> {
        self.intervals
            .iter()
            .find(|interval| interval.start <= time && time < interval.end)
    }
}
