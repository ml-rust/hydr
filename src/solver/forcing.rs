use super::{BoundaryConditions, BoundaryForcing, EngineError};
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
    /// Potential evaporation in m/s, capped by water remaining after infiltration.
    pub potential_evaporation: Vec<f64>,
    /// Prescribed land-loss demand in m/s. Removes proportional donor momentum.
    ///
    /// This is no infiltration model. Unmet demand rejects a trial without capping demand.
    pub withdrawal: Vec<f64>,
    /// Physical or source-complete data for each declared open face.
    pub boundaries: Vec<BoundaryForcing>,
}

/// Checked forcing coverage with no gaps, implicit rain, or provider callbacks.
#[derive(Clone, Debug, PartialEq)]
pub struct ForcingSchedule {
    intervals: Vec<ForcingInterval>,
}

impl ForcingSchedule {
    /// Checks spatial arrays, contiguous interval support, and complete open-face assignments.
    ///
    /// # Errors
    /// Rejects gaps, overlap, nonfinite inputs, negative rates, or missing/duplicate open faces.
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
        let declared: BTreeSet<_> = boundaries.open_faces().iter().copied().collect();
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
                ("potential_evaporation", &interval.potential_evaporation),
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
            if interval.boundaries.len() != declared.len() {
                return Err(EngineError::rejected(
                    "forcing",
                    format!(
                        "interval {index}: supply exactly one condition for each declared open face"
                    ),
                ));
            }
            let mut seen = BTreeSet::new();
            for value in &interval.boundaries {
                let face = value.face();
                if !declared.contains(&face) || !seen.insert(face) {
                    return Err(EngineError::rejected(
                        "forcing",
                        format!(
                            "interval {index}, face={face:?}: supply unique declared open-face support"
                        ),
                    ));
                }
                check_boundary(domain, value, index)?;
            }
        }
        Ok(Self { intervals })
    }

    /// Returns immutable forcing intervals.
    pub fn intervals(&self) -> &[ForcingInterval] {
        &self.intervals
    }

    /// Returns the checked contiguous tail from interval `from`, or `None` when it is empty.
    pub(super) fn window(&self, from: usize) -> Option<Self> {
        let intervals = self.intervals.get(from..)?;
        (!intervals.is_empty()).then(|| Self {
            intervals: intervals.to_vec(),
        })
    }

    pub(super) fn at(&self, time: f64) -> Option<&ForcingInterval> {
        self.intervals
            .iter()
            .find(|interval| interval.start <= time && time < interval.end)
    }
}

fn check_boundary(
    domain: &Domain,
    value: &BoundaryForcing,
    interval: usize,
) -> Result<(), EngineError> {
    let face = value.face();
    let cell = super::boundary::exterior_cell(domain, face)?;
    let valid = match value {
        BoundaryForcing::Flux(port) => {
            port.water_rate.is_finite()
                && port.momentum_rate.iter().all(|value| value.is_finite())
                && port.signal_speed.is_finite()
                && port.signal_speed > 0.0
                && !port.provenance.trim().is_empty()
        }
        BoundaryForcing::Stage(stage) => {
            stage.elevation.is_finite()
                && (stage.elevation - domain.bed()[cell]).is_finite()
                && stage.elevation > domain.bed()[cell]
                && stage.inflow_tangential_velocity.is_none_or(f64::is_finite)
        }
        BoundaryForcing::Discharge(discharge) => {
            discharge.outward_rate.is_finite()
                && discharge
                    .inflow_tangential_velocity
                    .is_none_or(f64::is_finite)
                && (discharge.outward_rate >= 0.0 || discharge.inflow_tangential_velocity.is_some())
        }
        BoundaryForcing::Exterior(exterior) => {
            exterior.conserved(domain)?;
            true
        }
        BoundaryForcing::Transmissive(_) => true,
    };
    if !valid {
        return Err(EngineError::rejected(
            "forcing",
            format!(
                "interval {interval}, boundary={value:?}: supply finite admitted data, positive wet stage depth, and an incoming tangent for prescribed inflow"
            ),
        ));
    }
    Ok(())
}
