//! Pool-end receipts of one interval and their lifetime counters, host resident.

use crate::coupling::surface_channels::SurfaceChannelError;
use crate::coupling::surface_channels::ports::two_sum;
use crate::numerics::outward::upward_add;

/// Pool-end receipts of one accepted interval, in ascending pool-end identity.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in super::super) struct PoolEndInterval {
    /// Signed volume of every end, positive from the reach into the pool, in m³.
    pub volume_m3: Vec<f64>,
    /// Reservoir head of every end: its pool's temporal midpoint stage, in m.
    pub head_m: Vec<f64>,
}

impl PoolEndInterval {
    /// Summed signed volume of every end, positive into the pools, and the summed magnitude.
    pub fn totals(&self) -> (f64, f64) {
        (
            self.volume_m3.iter().sum(),
            self.volume_m3.iter().map(|v| v.abs()).sum(),
        )
    }
}

/// Lifetime pool-end volumes in ascending pool-end identity.
#[derive(Clone, Debug, Default, PartialEq)]
pub(in super::super) struct PoolEndCounters {
    /// Water each reach gave its pool, in m³.
    pub to_pool_m3: Vec<f64>,
    /// Water each pool gave its reach, in m³.
    pub to_reach_m3: Vec<f64>,
    /// Upward-rounded sum of every addition error of both counters, in m³.
    pub roundoff_bound_m3: Vec<f64>,
}

impl PoolEndCounters {
    /// Zero counters for `ends` pool ends.
    pub fn zero(ends: usize) -> Self {
        Self {
            to_pool_m3: vec![0.0; ends],
            to_reach_m3: vec![0.0; ends],
            roundoff_bound_m3: vec![0.0; ends],
        }
    }

    /// True without pool ends.
    pub fn is_empty(&self) -> bool {
        self.to_pool_m3.is_empty()
    }

    /// Lifetime totals over every end: into the reaches, out of them, and the rounding bound.
    pub fn reach_totals(&self) -> (f64, f64, f64) {
        (
            self.to_reach_m3.iter().sum(),
            self.to_pool_m3.iter().sum(),
            self.roundoff_bound_m3.iter().sum(),
        )
    }

    /// Adds one interval's pool-end volumes with error-free addition bounds.
    ///
    /// Counters without pool ends stay `+0.0`.
    pub fn advance(&self, interval: &PoolEndInterval) -> Result<Self, SurfaceChannelError> {
        if self.is_empty() {
            return Ok(self.clone());
        }
        if interval.volume_m3.len() != self.to_pool_m3.len() {
            return Err(SurfaceChannelError::Numerical(
                "pool-end counters differ from the pool ends of the interval".into(),
            ));
        }
        let numerical = |error: crate::numerics::outward::OutwardError| {
            SurfaceChannelError::Numerical(format!("pool-end counters: {error}"))
        };
        let mut next = self.clone();
        for (((to_pool, to_reach), bound), &signed) in next
            .to_pool_m3
            .iter_mut()
            .zip(next.to_reach_m3.iter_mut())
            .zip(next.roundoff_bound_m3.iter_mut())
            .zip(&interval.volume_m3)
        {
            let (forward, forward_low) = two_sum(*to_pool, signed.max(0.0));
            let (backward, backward_low) = two_sum(*to_reach, (-signed).max(0.0));
            *to_pool = forward;
            *to_reach = backward;
            if forward_low != 0.0 || backward_low != 0.0 {
                let low = upward_add(forward_low.abs(), backward_low.abs()).map_err(numerical)?;
                *bound = upward_add(*bound, low).map_err(numerical)?;
            }
        }
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_split_each_direction_and_stay_zero_without_ends() -> Result<(), SurfaceChannelError>
    {
        let empty = PoolEndCounters::zero(0);
        assert_eq!(empty.advance(&PoolEndInterval::default())?, empty);
        let counters = PoolEndCounters::zero(2);
        let interval = PoolEndInterval {
            volume_m3: vec![0.25, -0.5],
            head_m: vec![1.0, 1.0],
        };
        let next = counters.advance(&interval)?.advance(&interval)?;
        assert_eq!(next.to_pool_m3, [0.5, 0.0]);
        assert_eq!(next.to_reach_m3, [0.0, 1.0]);
        assert_eq!(next.reach_totals(), (1.0, 0.5, 0.0));
        assert_eq!(interval.totals(), (-0.25, 0.75));
        Ok(())
    }
}
