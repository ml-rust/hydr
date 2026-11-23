use super::definition::Definition;
use super::forcing_history::{self, ForcingPrefix};
use super::{
    BoundaryConditions, BoundaryForcing, EngineError, ExteriorState, ForcingInterval, SolverConfig,
};
use crate::model::Domain;
use crate::soil::{LandCell, LandModel};

/// Restart source compared against an engine definition.
pub(super) struct Candidate<'a> {
    pub domain: &'a Domain,
    pub land: &'a LandModel,
    pub boundaries: &'a BoundaryConditions,
    pub config: &'a SolverConfig,
    pub prefix: &'a ForcingPrefix,
    pub forcing: &'a [ForcingInterval],
    /// Cached digest chain over `forcing`, when the source holds one.
    pub chain: Option<&'a [[u8; 32]]>,
    /// Absolute index of the first interval the restart source needs.
    pub retained_from: usize,
}

impl<'a> Candidate<'a> {
    /// Describes an in-memory definition whose restart needs intervals from local `retained`.
    pub(super) fn of(definition: &'a Definition, retained: usize) -> Self {
        Self {
            domain: &definition.domain,
            land: &definition.land,
            boundaries: &definition.boundaries,
            config: &definition.config,
            prefix: &definition.prefix,
            forcing: definition.forcing.intervals(),
            chain: Some(&definition.chain),
            retained_from: definition.prefix.intervals.saturating_add(retained),
        }
    }
}

impl Definition {
    /// Compares static data, then forcing histories folded to the later prefix base.
    ///
    /// Folded prefixes compare count, start bits, end bits, and digest.
    /// Remaining intervals compare bitwise. Numerical identity includes signed zero.
    ///
    /// # Errors
    /// Rejects an engine whose retained window starts after the interval the source needs.
    pub(super) fn equivalent(&self, other: &Candidate<'_>) -> Result<bool, EngineError> {
        self.equivalent_inner(other, true)
    }

    pub(super) fn equivalent_ignoring_bed(
        &self,
        other: &Candidate<'_>,
    ) -> Result<bool, EngineError> {
        self.equivalent_inner(other, false)
    }

    fn equivalent_inner(
        &self,
        other: &Candidate<'_>,
        check_bed: bool,
    ) -> Result<bool, EngineError> {
        if self.prefix.intervals > other.retained_from {
            return Err(EngineError::rejected(
                "forcing window",
                format!(
                    "engine forcing starts at interval {}, checkpoint needs interval {} onward: build the engine with forcing that starts at interval {} or earlier",
                    self.prefix.intervals, other.retained_from, other.retained_from
                ),
            ));
        }
        if !self.same_static(other, check_bed) {
            return Ok(false);
        }
        let base = self.prefix.intervals.max(other.prefix.intervals);
        let own = base - self.prefix.intervals;
        let theirs = base - other.prefix.intervals;
        let (Some(left), Some(right)) = (
            self.folded(own),
            forcing_history::folded(other.prefix, other.forcing, other.chain, theirs),
        ) else {
            return Ok(false);
        };
        if left.intervals != right.intervals
            || !same_bits(&[left.start, left.end], &[right.start, right.end])
            || left.digest != right.digest
        {
            return Ok(false);
        }
        let (Some(left), Some(right)) = (
            self.forcing.intervals().get(own..),
            other.forcing.get(theirs..),
        ) else {
            return Ok(false);
        };
        Ok(left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| same_interval(left, right)))
    }

    fn same_static(&self, other: &Candidate<'_>, check_bed: bool) -> bool {
        let (domain, land, config) = (other.domain, other.land, other.config);
        let same_domain = if check_bed {
            self.domain == *domain
        } else {
            self.domain.grid() == domain.grid()
                && self.domain.manning() == domain.manning()
                && self.domain.active() == domain.active()
                && self.domain.reference() == domain.reference()
        };
        if !same_domain
            || self.land != *land
            || self.boundaries != *other.boundaries
            || self.config != *config
        {
            return false;
        }
        let geometry = |domain: &Domain| {
            let (dx, dy) = domain.grid().spacing();
            let [x, y] = domain.reference().origin();
            [dx, dy, x, y]
        };
        if (check_bed && !same_bits(self.domain.bed(), domain.bed()))
            || !same_bits(self.domain.manning(), domain.manning())
            || !same_bits(&geometry(&self.domain), &geometry(domain))
            || !same_bits(&controls(&self.config), &controls(config))
        {
            return false;
        }
        self.land
            .cells()
            .iter()
            .zip(land.cells())
            .all(|(a, b)| match (a, b) {
                (LandCell::Impervious, LandCell::Impervious) => true,
                (LandCell::ModifiedGreenAmpt(a), LandCell::ModifiedGreenAmpt(b)) => same_bits(
                    &[
                        a.suction_ft(),
                        a.conductivity_ft_per_s(),
                        a.maximum_deficit(),
                        a.upper_zone_depth_ft(),
                        a.capacity_ft(),
                        a.recovered_deficit_ceiling(),
                    ],
                    &[
                        b.suction_ft(),
                        b.conductivity_ft_per_s(),
                        b.maximum_deficit(),
                        b.upper_zone_depth_ft(),
                        b.capacity_ft(),
                        b.recovered_deficit_ceiling(),
                    ],
                ),
                _ => false,
            })
    }
}

/// Compares one forcing interval bit for bit, including faces, variants, provenance, and signed zero.
pub(super) fn same_interval(left: &ForcingInterval, right: &ForcingInterval) -> bool {
    left == right && interval_bits(left, right)
}

fn interval_bits(left: &ForcingInterval, right: &ForcingInterval) -> bool {
    same_bits(&[left.start, left.end], &[right.start, right.end])
        && same_bits(&left.rain, &right.rain)
        && same_bits(&left.withdrawal, &right.withdrawal)
        && same_bits(&left.potential_evaporation, &right.potential_evaporation)
        && left.boundaries.len() == right.boundaries.len()
        && left
            .boundaries
            .iter()
            .zip(&right.boundaries)
            .all(|(left, right)| boundary_bits(left, right))
}

pub(super) fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn controls(config: &SolverConfig) -> [f64; 14] {
    [
        config.gravity,
        config.cfl,
        config.max_step,
        config.min_step,
        config.root_absolute_depth,
        config.root_relative_depth,
        config.volume_absolute,
        config.volume_relative,
        config.momentum_film_depth,
        config.max_cumulative_water_adjustment_m3,
        config.boundary.discharge_absolute,
        config.boundary.discharge_relative,
        config.boundary.characteristic_relative,
        config.boundary.critical_relative,
    ]
}

fn boundary_bits(left: &BoundaryForcing, right: &BoundaryForcing) -> bool {
    let optional = |left: Option<f64>, right: Option<f64>| match (left, right) {
        (Some(left), Some(right)) => left.to_bits() == right.to_bits(),
        (None, None) => true,
        _ => false,
    };
    match (left, right) {
        (BoundaryForcing::Flux(left), BoundaryForcing::Flux(right)) => same_bits(
            &[
                left.water_rate,
                left.momentum_rate[0],
                left.momentum_rate[1],
                left.signal_speed,
            ],
            &[
                right.water_rate,
                right.momentum_rate[0],
                right.momentum_rate[1],
                right.signal_speed,
            ],
        ),
        (BoundaryForcing::Stage(left), BoundaryForcing::Stage(right)) => {
            left.elevation.to_bits() == right.elevation.to_bits()
                && optional(
                    left.inflow_tangential_velocity,
                    right.inflow_tangential_velocity,
                )
        }
        (BoundaryForcing::Discharge(left), BoundaryForcing::Discharge(right)) => {
            left.outward_rate.to_bits() == right.outward_rate.to_bits()
                && optional(
                    left.inflow_tangential_velocity,
                    right.inflow_tangential_velocity,
                )
        }
        (BoundaryForcing::Exterior(left), BoundaryForcing::Exterior(right)) => {
            match (&left.state, &right.state) {
                (ExteriorState::Dry, ExteriorState::Dry) => true,
                (
                    ExteriorState::Wet {
                        elevation: a,
                        outward_velocity: b,
                        tangential_velocity: c,
                    },
                    ExteriorState::Wet {
                        elevation: x,
                        outward_velocity: y,
                        tangential_velocity: z,
                    },
                ) => same_bits(&[*a, *b, *c], &[*x, *y, *z]),
                _ => false,
            }
        }
        (BoundaryForcing::Transmissive(left), BoundaryForcing::Transmissive(right)) => {
            left == right
        }
        _ => false,
    }
}
