use super::{ChannelConfig, ChannelError};
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

/// Explicit rectangular geometry and numerical controls for one unbranched reach.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelReachDefinition {
    /// Common vertical reference for every bed and returned water stage.
    pub vertical_datum: String,
    /// Evidence supporting the geometry and rectangular-flow assumptions.
    pub provenance: String,
    /// Nonempty connected cells in increasing longitudinal order.
    pub cells: Vec<RectangularChannelCell>,
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

/// Endpoint policy using the adjacent endpoint cell's width and bed elevation.
#[derive(Clone, Debug, PartialEq)]
pub enum ChannelBoundary {
    /// Genuine impermeable wall, implemented by reflecting longitudinal velocity.
    Wall,
    /// Held exterior reservoir state, without prescribing achieved discharge.
    Exterior {
        /// Exterior water depth above the endpoint cell bed in metres.
        depth_m: f64,
        /// Exterior velocity in the reach's increasing direction in m/s.
        velocity_m_per_s: f64,
    },
}

/// Complete held endpoint states over one bounded interval, without lateral sources.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelBoundaryInterval {
    /// Exact accepted start time in seconds.
    pub start_s: f64,
    /// End of endpoint-state support in seconds.
    pub end_s: f64,
    /// Policy at the upstream longitudinal endpoint.
    pub left: ChannelBoundary,
    /// Policy at the downstream longitudinal endpoint.
    pub right: ChannelBoundary,
}

impl ChannelReachDefinition {
    pub(super) fn check(&self) -> Result<(), ChannelError> {
        self.config.check()?;
        for (name, value) in [
            ("vertical_datum", &self.vertical_datum),
            ("provenance", &self.provenance),
        ] {
            check_text(name, value)?;
        }
        if self.cells.is_empty() {
            return Err(ChannelError::input(
                "cell count=0: supply at least one explicit rectangular cell",
            ));
        }
        let mut ids = BTreeSet::new();
        let mut length = 0.0;
        for (index, cell) in self.cells.iter().enumerate() {
            check_text("id", &cell.id).map_err(|error| error.at(index))?;
            if !ids.insert(&cell.id) {
                return Err(ChannelError::input(format!(
                    "id={:?}: supply a unique cell identity",
                    cell.id
                ))
                .at(index));
            }
            for (name, value) in [
                ("length_m", cell.length_m),
                ("width_m", cell.width_m),
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
            let plan_area = cell.width_m * cell.length_m;
            let maximum_volume = plan_area * cell.maximum_depth_m;
            let next_length = length + cell.length_m;
            if !cell.bed_elevation_m.is_finite()
                || !maximum_stage.is_finite()
                || maximum_stage <= cell.bed_elevation_m
                || !plan_area.is_finite()
                || plan_area <= 0.0
                || !maximum_volume.is_finite()
                || maximum_volume <= 0.0
                || !next_length.is_finite()
                || next_length <= length
                || !cell.manning_n.is_finite()
                || cell.manning_n < 0.0
            {
                return Err(ChannelError::input(format!(
                    "id={:?}, bed={}, maximum_stage={maximum_stage}, plan_area={plan_area}, maximum_volume={maximum_volume}, reach_length={next_length}, manning_n={}: supply representable geometry and finite nonnegative roughness",
                    cell.id, cell.bed_elevation_m, cell.manning_n
                ))
                .at(index));
            }
            length = next_length;
        }
        Ok(())
    }
}

impl ChannelBoundaryInterval {
    pub(super) fn check(
        &self,
        time_s: f64,
        requested_end_s: f64,
        definition: &ChannelReachDefinition,
    ) -> Result<(), ChannelError> {
        if self.start_s.to_bits() != time_s.to_bits()
            || !self.end_s.is_finite()
            || self.end_s <= self.start_s
            || !requested_end_s.is_finite()
            || requested_end_s <= time_s
            || requested_end_s > self.end_s
        {
            return Err(ChannelError::input(format!(
                "accepted_time={time_s}, forcing_span=[{},{}], requested_end={requested_end_s}: supply the exact accepted start and a later endpoint within forcing support",
                self.start_s, self.end_s
            )));
        }
        let last = definition.cells.len().checked_sub(1).ok_or_else(|| {
            ChannelError::input("cell count=0: construct a nonempty rectangular reach")
        })?;
        for (boundary, index) in [(&self.left, 0), (&self.right, last)] {
            let cell = &definition.cells[index];
            if let ChannelBoundary::Exterior {
                depth_m,
                velocity_m_per_s,
            } = boundary
            {
                let unit_discharge = depth_m * velocity_m_per_s;
                let discharge = unit_discharge * cell.width_m;
                if !depth_m.is_finite()
                    || *depth_m < 0.0
                    || *depth_m > cell.maximum_depth_m
                    || !velocity_m_per_s.is_finite()
                    || (*depth_m == 0.0 && *velocity_m_per_s != 0.0)
                    || !unit_discharge.is_finite()
                    || !discharge.is_finite()
                    || (*depth_m > 0.0
                        && *velocity_m_per_s != 0.0
                        && (unit_discharge == 0.0 || discharge == 0.0))
                {
                    return Err(ChannelError::input(format!(
                        "exterior depth={depth_m}, velocity={velocity_m_per_s}, maximum_depth={}: supply supported finite depth and velocity, with zero velocity when dry",
                        cell.maximum_depth_m
                    ))
                    .at(index));
                }
            }
        }
        Ok(())
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
