//! Explicit complete antecedent and per-destination source input.

use super::{Subarea, SubcatchmentError, SubcatchmentParameters};
use crate::soil::NativeSoilState;

/// Identity and complete physical support for one standalone component.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentDefinition {
    /// Unique nonempty identity without control characters. No GIS disjointness is implied.
    pub id: String,
    /// Checked geometry, soil, and routing.
    pub parameters: SubcatchmentParameters,
}

/// Complete accepted physical memory, without a previous-rate routing queue.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubcatchmentState {
    /// Ordered subarea depths in metres. Absent support requires zero.
    pub depths_m: [f64; 3],
    /// Complete native soil memory exactly when pervious area is positive.
    pub soil: Option<NativeSoilState>,
}
impl SubcatchmentState {
    /// Explicitly select zero surface water and original initialized pervious soil memory.
    pub fn initialized(parameters: &SubcatchmentParameters) -> Result<Self, SubcatchmentError> {
        let soil = parameters.soil_parameters().map(|p| NativeSoilState {
            imd: p.maximum_deficit(),
            f_ft: 0.0,
            fu_ft: 0.0,
            saturated: false,
            t_s: 0.0,
        });
        let state = Self {
            depths_m: [0.0; 3],
            soil,
        };
        state.check(parameters)?;
        Ok(state)
    }
    pub(crate) fn check(
        &self,
        parameters: &SubcatchmentParameters,
    ) -> Result<(), SubcatchmentError> {
        for area in Subarea::ALL {
            let depth = self.depths_m[area.index()];
            if !depth.is_finite()
                || depth < 0.0
                || (parameters.area_m2(area) == 0.0 && depth != 0.0)
            {
                return Err(SubcatchmentError::input(format!("depth_m={depth}: require finite nonnegative active storage and zero absent depth")).at_subarea(area));
            }
        }
        match (self.soil, parameters.soil_parameters()) {
            (None, None) => Ok(()),
            (Some(state), Some(parameters)) => state
                .check(parameters, 0)
                .map_err(|error| SubcatchmentError::from(error).at_subarea(Subarea::Pervious)),
            _ => Err(SubcatchmentError::input(
                "complete native soil state must match positive pervious support",
            )
            .at_subarea(Subarea::Pervious)),
        }
    }
}

/// Constant rain/evaporation with authoritative destination volumes over the original interval.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SubcatchmentInput {
    /// Nonnegative shared rainfall rate in m/s, applied once over represented support.
    pub rain_rate_m_per_s: f64,
    /// Nonnegative potential evaporation rate in m/s, capped after infiltration.
    pub potential_evaporation_m_per_s: f64,
    /// Authoritative external runon volumes in m³ for the three explicit destinations.
    pub external_runon_volumes_m3: [f64; 3],
}
