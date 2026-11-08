use super::engine::Definition;
use super::{BoundaryConditions, ForcingSchedule, SolverConfig};
use crate::model::Domain;

impl Definition {
    /// Numerical identity includes signed zero. Mathematical equality alone cannot promise bitwise continuation.
    pub(super) fn matches(
        &self,
        domain: &Domain,
        boundaries: &BoundaryConditions,
        forcing: &ForcingSchedule,
        config: &SolverConfig,
    ) -> bool {
        if self.domain != *domain
            || self.boundaries != *boundaries
            || self.forcing != *forcing
            || self.config != *config
        {
            return false;
        }
        let geometry = |domain: &Domain| {
            let (dx, dy) = domain.grid().spacing();
            let [x, y] = domain.reference().origin();
            [dx, dy, x, y]
        };
        if !same_bits(self.domain.bed(), domain.bed())
            || !same_bits(self.domain.manning(), domain.manning())
            || !same_bits(&geometry(&self.domain), &geometry(domain))
            || !same_bits(&controls(&self.config), &controls(config))
        {
            return false;
        }
        self.forcing
            .intervals()
            .iter()
            .zip(forcing.intervals())
            .all(|(left, right)| {
                same_bits(&[left.start, left.end], &[right.start, right.end])
                    && same_bits(&left.rain, &right.rain)
                    && same_bits(&left.withdrawal, &right.withdrawal)
                    && left.ports.iter().zip(&right.ports).all(|(left, right)| {
                        same_bits(
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
                        )
                    })
            })
    }
}

fn same_bits(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn controls(config: &SolverConfig) -> [f64; 9] {
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
    ]
}
