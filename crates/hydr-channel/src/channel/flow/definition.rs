pub use super::geometry::{ChannelGeometry, TriangularChannelCell};
use super::{ChannelConfig, ChannelError};
use crate::forcing::WaterAmount;
use std::collections::BTreeSet;

/// One constant rectangular cell, ordered along the reach's increasing longitudinal coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct RectangularChannelCell {
    /// Unique stable cell identity.
    pub id: String,
    /// Longitudinal cell length in metres.
    pub length_m: f64,
    /// Constant clear channel width in metres.
    pub width_m: f64,
    /// Bed elevation in the reach's declared vertical datum, in metres.
    pub bed_elevation_m: f64,
    /// Highest supported depth in metres, without bank extrapolation.
    pub maximum_depth_m: f64,
    /// Manning roughness in s/m^(1/3), including zero for frictionless flow.
    pub manning_n: f64,
}

/// Declared geometry and numerical controls for one unbranched reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelReachDefinition {
    /// Common vertical reference for every bed and returned water stage.
    pub vertical_datum: String,
    /// Evidence supporting the geometry and flow assumptions.
    pub provenance: String,
    /// Nonempty connected cells in increasing longitudinal order.
    pub geometry: ChannelGeometry,
    /// Selected lateral sources and withdrawals; empty explicitly selects an unforced reach.
    pub lateral_bindings: Vec<ChannelLateralBinding>,
    /// Dimensional numerical controls.
    pub config: ChannelConfig,
}

/// Authoritative initial cell volumes and integrated longitudinal momenta.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelInitialState {
    /// Initial elapsed time in seconds.
    pub time_s: f64,
    /// Water volume per ordered cell in m³.
    pub volume_m3: Vec<f64>,
    /// Density-normalized longitudinal momentum per cell in m⁴/s: discharge times cell length.
    pub longitudinal_momentum_m4_per_s: Vec<f64>,
}

/// Endpoint policy using the declared section and adjacent bed elevation.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelBoundary {
    /// Held water elevation in the reach datum.
    Stage {
        /// Water elevation in metres.
        elevation_m: f64,
    },
    /// Held full-section discharge, positive along the increasing longitudinal coordinate.
    Discharge {
        /// Original discharge operand in m³/s.
        discharge_m3_per_s: f64,
    },
    /// Held quiescent reservoir with complete exit mixing and explicit entrance loss.
    ///
    /// The interior and delivered trace remain strictly wet and subcritical.
    /// Zero entrance loss selects an ideal entrance. This policy supplies no finite reservoir storage.
    QuiescentReservoir {
        /// Finite reservoir water elevation in metres in the reach datum.
        elevation_m: f64,
        /// Finite nonnegative dimensionless entrance loss referenced to incoming channel velocity head.
        entrance_loss_coefficient: f64,
    },
    /// Genuine impermeable wall, implemented by reflecting longitudinal velocity.
    Wall,
    /// Held exterior reservoir state, without prescribing achieved discharge.
    Exterior {
        /// Exterior water depth above the endpoint cell bed in metres.
        depth_m: f64,
        /// Exterior velocity in the reach's increasing direction in m/s.
        velocity_m_per_s: f64,
    },
    /// Direct water and momentum rates from a flux provider, inserted at the end face.
    ///
    /// Rectangular and triangular reaches admit it. Surveyed reaches reject it.
    FluxPort(super::ChannelFluxPort),
}

/// Fixed physical direction of a declared lateral boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelLateralDirection {
    /// Supplied water enters the receiving cell with explicit longitudinal velocity.
    Inflow,
    /// Water leaves the donating cell at its accepted-start longitudinal velocity.
    Withdrawal,
}

/// One selected, nonduplicated lateral input attached to an existing reach cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChannelLateralBinding {
    /// Unique stable lateral boundary identity.
    pub boundary_id: String,
    /// Exact receiving or donating cell identity.
    pub cell_id: String,
    /// Fixed direction accepted at this boundary.
    pub direction: ChannelLateralDirection,
    /// Evidence excluding duplicate supplied water; this establishes no geographic ownership proof.
    pub ownership_assertion: String,
}

/// Original lateral water input with direction-specific momentum semantics.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelLateralInput {
    /// Water enters with the caller's explicitly supplied longitudinal velocity.
    Inflow {
        /// Exact declared lateral boundary identity.
        boundary_id: String,
        /// Original nonnegative rate or integrated volume, retaining its variant and bits.
        amount: WaterAmount,
        /// Incoming longitudinal velocity in the reach's increasing direction, in m/s.
        longitudinal_velocity_m_per_s: f64,
    },
    /// Water leaves at the accepted-start donor velocity, without a caller velocity.
    Withdrawal {
        /// Exact declared lateral boundary identity.
        boundary_id: String,
        /// Original nonnegative rate or integrated volume, retaining its variant and bits.
        amount: WaterAmount,
    },
}

/// Complete held endpoint states and lateral inputs over one bounded interval.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelForcingInterval {
    /// Exact accepted start time in seconds.
    pub start_s: f64,
    /// End of endpoint-state support in seconds.
    pub end_s: f64,
    /// Policy at the upstream longitudinal endpoint.
    pub left: ChannelBoundary,
    /// Policy at the downstream longitudinal endpoint.
    pub right: ChannelBoundary,
    /// One explicit input per declared lateral binding, including zeros.
    pub lateral_inputs: Vec<ChannelLateralInput>,
}

impl ChannelReachDefinition {
    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        let same_values = |left: &[f64], right: &[f64]| {
            left.len() == right.len()
                && left
                    .iter()
                    .zip(right)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
        };
        let left = &self.config;
        let right = &other.config;
        self.vertical_datum == other.vertical_datum
            && self.provenance == other.provenance
            && self.geometry.same_bits(&other.geometry)
            && left.boundary.same_bits(&right.boundary)
            && left.max_retries == right.max_retries
            && left.root_iterations == right.root_iterations
            && left.triangular_reconstruction == right.triangular_reconstruction
            && same_values(
                &[
                    left.gravity,
                    left.cfl,
                    left.max_step,
                    left.min_step,
                    left.root_absolute_depth,
                    left.root_relative_depth,
                    left.riemann_water_flux_absolute_m3_per_s,
                    left.riemann_momentum_flux_absolute_m4_per_s2,
                    left.riemann_flux_relative,
                    left.volume_absolute,
                    left.volume_relative,
                    left.momentum_absolute,
                    left.momentum_relative,
                ],
                &[
                    right.gravity,
                    right.cfl,
                    right.max_step,
                    right.min_step,
                    right.root_absolute_depth,
                    right.root_relative_depth,
                    right.riemann_water_flux_absolute_m3_per_s,
                    right.riemann_momentum_flux_absolute_m4_per_s2,
                    right.riemann_flux_relative,
                    right.volume_absolute,
                    right.volume_relative,
                    right.momentum_absolute,
                    right.momentum_relative,
                ],
            )
            && self.lateral_bindings == other.lateral_bindings
    }

    /// `coordinator_bindings` admits lateral identities with the reserved surface-exchange prefix.
    pub(super) fn check(&self, coordinator_bindings: bool) -> Result<(), ChannelError> {
        self.config.check()?;
        for (name, value) in [
            ("vertical_datum", &self.vertical_datum),
            ("provenance", &self.provenance),
        ] {
            check_text(name, value)?;
        }
        if self.geometry.len() == 0 {
            return Err(ChannelError::input(
                "cell count=0: supply at least one explicit channel cell",
            ));
        }
        self.geometry.check()?;
        if let super::ChannelGeometry::PrismaticSurveyed { profile, .. } = &self.geometry
            && profile.vertical_datum != self.vertical_datum
        {
            return Err(ChannelError::input(format!(
                "profile datum={:?}, reach datum={:?}: supply one shared vertical datum",
                profile.vertical_datum, self.vertical_datum
            )));
        }
        let mut ids = BTreeSet::new();
        let mut length = 0.0;
        for (index, cell) in self.geometry.cells().enumerate() {
            check_text("id", cell.id).map_err(|error| error.at(index))?;
            if !ids.insert(cell.id) {
                return Err(ChannelError::input(format!(
                    "id={:?}: supply a unique cell identity",
                    cell.id
                ))
                .at(index));
            }
            for (name, value) in [
                ("length_m", cell.length_m),
                ("maximum_depth_m", cell.maximum_depth_m),
            ] {
                if !value.is_finite() || value <= 0.0 {
                    return Err(ChannelError::input(format!(
                        "{name}={value}: supply a finite positive dimension"
                    ))
                    .at(index));
                }
            }
            let maximum_stage = cell.bed_elevation_m + cell.maximum_depth_m;
            let maximum_volume = self.geometry.analytic_maximum_volume(index);
            let next_length = length + cell.length_m;
            if !cell.bed_elevation_m.is_finite()
                || !maximum_stage.is_finite()
                || maximum_stage <= cell.bed_elevation_m
                || maximum_volume.is_some_and(|value| !value.is_finite() || value <= 0.0)
                || !next_length.is_finite()
                || next_length <= length
                || !cell.manning_n.is_finite()
                || cell.manning_n < 0.0
            {
                return Err(ChannelError::input(format!(
                    "id={:?}, bed={}, maximum_stage={maximum_stage}, maximum_volume={maximum_volume:?}, reach_length={next_length}, manning_n={}: supply representable geometry and finite nonnegative roughness",
                    cell.id, cell.bed_elevation_m, cell.manning_n
                ))
                .at(index));
            }
            length = next_length;
        }
        let mut boundaries = BTreeSet::new();
        for binding in &self.lateral_bindings {
            check_text("boundary_id", &binding.boundary_id)?;
            if !coordinator_bindings {
                super::coupling::reject_reserved_boundary(&binding.boundary_id)?;
            }
            check_text("cell_id", &binding.cell_id)?;
            check_text("ownership_assertion", &binding.ownership_assertion)?;
            if !boundaries.insert(binding.boundary_id.as_str()) {
                return Err(ChannelError::input(format!(
                    "boundary_id={:?}: supply a unique lateral boundary identity",
                    binding.boundary_id
                )));
            }
            if !ids.contains(binding.cell_id.as_str()) {
                return Err(ChannelError::input(format!(
                    "boundary_id={:?}, cell_id={:?}: attach the lateral boundary to an existing cell",
                    binding.boundary_id, binding.cell_id
                )));
            }
        }
        Ok(())
    }
}

pub(super) fn check_forcing_interval(
    time_s: f64,
    requested_end_s: f64,
    start_s: f64,
    end_s: f64,
) -> Result<(), ChannelError> {
    if !start_s.is_finite()
        || start_s.to_bits() != time_s.to_bits()
        || !end_s.is_finite()
        || end_s <= start_s
        || !requested_end_s.is_finite()
        || requested_end_s <= time_s
        || requested_end_s > end_s
    {
        return Err(ChannelError::input(format!(
            "accepted_time={time_s}, forcing_span=[{start_s},{end_s}], requested_end={requested_end_s}: supply the exact accepted start and a later endpoint within forcing support"
        )));
    }
    Ok(())
}

impl ChannelLateralInput {
    pub(crate) fn boundary_id(&self) -> &str {
        match self {
            Self::Inflow { boundary_id, .. } | Self::Withdrawal { boundary_id, .. } => boundary_id,
        }
    }

    pub(crate) fn amount(&self) -> WaterAmount {
        match self {
            Self::Inflow { amount, .. } | Self::Withdrawal { amount, .. } => *amount,
        }
    }

    #[doc(hidden)]
    pub fn same_bits(&self, other: &Self) -> bool {
        self.boundary_id() == other.boundary_id()
            && self.amount().same_bits(&other.amount())
            && match (self, other) {
                (
                    Self::Inflow {
                        longitudinal_velocity_m_per_s: left,
                        ..
                    },
                    Self::Inflow {
                        longitudinal_velocity_m_per_s: right,
                        ..
                    },
                ) => left.to_bits() == right.to_bits(),
                (Self::Withdrawal { .. }, Self::Withdrawal { .. }) => true,
                _ => false,
            }
    }
}

fn check_text(name: &str, value: &str) -> Result<(), ChannelError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(ChannelError::input(format!(
            "{name}={value:?}: supply nonempty identity and evidence without control characters"
        )));
    }
    Ok(())
}
