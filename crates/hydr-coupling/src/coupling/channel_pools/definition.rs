use crate::channel::{ChannelInitialState, ChannelReachDefinition};
use crate::storage::{LevelPoolConfig, LevelPoolElement, LevelPoolState};

use super::ChannelPoolError;

/// Endpoint in the reach's increasing longitudinal coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelPoolEndpoint {
    /// First cell's left face.
    Left,
    /// Last cell's right face.
    Right,
}

/// Named unbranched reach with its native geometry and numerical controls.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolReach {
    /// Unique nonempty reach identity used by connections and forcing records.
    pub id: String,
    /// Native reach definition, including vertical datum and lateral bindings.
    pub definition: ChannelReachDefinition,
}

/// One channel endpoint connected to finite, approximately level pool storage.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolConnection {
    /// Unique nonempty connection identity used by transfer results and ledgers.
    pub id: String,
    /// Exact identity of the connected reach.
    pub reach_id: String,
    /// Connected endpoint, owned by this connection alone.
    pub endpoint: ChannelPoolEndpoint,
    /// Exact identity of the connected pool.
    pub pool_id: String,
    /// Finite nonnegative dimensionless entrance loss against incoming channel velocity head.
    pub entrance_loss_coefficient: f64,
    /// Evidence that the endpoint samples this pool at the channel's local bed elevation.
    pub physical_connection_assertion: String,
    /// Evidence supporting quiescent reservoir inflow and complete exit mixing into the pool.
    pub quiescent_reservoir_assertion: String,
}

/// Explicit convergence budgets and bounded work for common channel/pool intervals.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolConfig {
    /// Native individual-pool inverse-curve and volume accounting controls.
    pub storage: LevelPoolConfig,
    /// Absolute midpoint-head iteration allowance, in metres.
    pub head_absolute_m: f64,
    /// Dimensionless relative midpoint-head iteration allowance.
    pub head_relative: f64,
    /// Absolute signed-transfer iteration allowance, in cubic metres.
    pub transfer_absolute_m3: f64,
    /// Dimensionless relative signed-transfer iteration allowance.
    pub transfer_relative: f64,
    /// Absolute physical water-residual allowance, in cubic metres.
    pub mass_absolute_m3: f64,
    /// Dimensionless relative allowance against represented storage and external physical terms.
    pub mass_relative: f64,
    /// Smallest admitted common trial duration, in seconds.
    pub minimum_step_s: f64,
    /// Largest admitted common trial duration, in seconds.
    pub maximum_step_s: f64,
    /// Maximum fixed-point iterations per attempted common interval.
    pub iteration_limit: usize,
    /// Maximum complete interval recalculations after the initial attempt.
    pub retry_limit: usize,
}

impl ChannelPoolConfig {
    pub(super) fn check(&self) -> Result<(), ChannelPoolError> {
        self.storage.check()?;
        for (absolute_name, absolute, relative_name, relative) in [
            (
                "head_absolute_m",
                self.head_absolute_m,
                "head_relative",
                self.head_relative,
            ),
            (
                "transfer_absolute_m3",
                self.transfer_absolute_m3,
                "transfer_relative",
                self.transfer_relative,
            ),
            (
                "mass_absolute_m3",
                self.mass_absolute_m3,
                "mass_relative",
                self.mass_relative,
            ),
        ] {
            if !absolute.is_finite()
                || absolute < 0.0
                || !relative.is_finite()
                || relative < 0.0
                || (absolute == 0.0 && relative == 0.0)
            {
                return Err(ChannelPoolError::input(format!(
                    "{absolute_name}={absolute},{relative_name}={relative}: supply finite nonnegative tolerances with positive combined allowance"
                )));
            }
        }
        if !self.minimum_step_s.is_finite()
            || self.minimum_step_s <= 0.0
            || !self.maximum_step_s.is_finite()
            || self.maximum_step_s < self.minimum_step_s
        {
            return Err(ChannelPoolError::input(format!(
                "minimum_step_s={},maximum_step_s={}: supply finite positive durations with maximum_step_s >= minimum_step_s",
                self.minimum_step_s, self.maximum_step_s
            )));
        }
        if self.iteration_limit == 0 {
            return Err(ChannelPoolError::input(
                "iteration_limit=0: supply a positive iteration count",
            ));
        }
        Ok(())
    }
}

/// Immutable finite-storage topology with explicit physical support and common numerical controls.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolSetup {
    /// Nonempty native reach definitions in initial-state and child-output order.
    pub reaches: Vec<ChannelPoolReach>,
    /// Nonempty exchange-only pool definitions in initial-state and child-output order.
    pub pools: Vec<LevelPoolElement>,
    /// Nonempty endpoint ownership declarations in transfer and connection-ledger order.
    pub connections: Vec<ChannelPoolConnection>,
    /// Common vertical reference, matching every channel and pool datum.
    pub vertical_datum: String,
    /// Evidence excluding duplicated represented channel and pool storage.
    pub nonoverlap_assertion: String,
    /// Explicit dimensional tolerances, common duration limits, and work limits.
    pub config: ChannelPoolConfig,
}

/// Complete authoritative child states at one common accepted origin.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelPoolInitialState {
    /// Finite nonnegative common elapsed time, in seconds, matching every reach's time exactly.
    pub time_s: f64,
    /// Native channel cell volumes and momenta in setup reach order.
    pub reaches: Vec<ChannelInitialState>,
    /// Authoritative pool volumes in setup pool order.
    pub pools: Vec<LevelPoolState>,
}
