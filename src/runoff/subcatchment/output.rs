//! Explicit supported-class output derived only at an application boundary.

use super::{SubcatchmentError, SubcatchmentReceipt};

/// Physical interval output of one represented class.
#[derive(Clone, Debug, PartialEq)]
pub struct SubareaOutput {
    /// Accepted depth in metres.
    pub depth_m: f64,
    /// Accepted storage in m³.
    pub storage_m3: f64,
    /// Applied rainfall in m³.
    pub rain_volume_m3: f64,
    /// Original external runon in m³.
    pub external_runon_volume_m3: f64,
    /// Authoritative internal received volume in m³.
    pub internal_received_volume_m3: f64,
    /// Original soil infiltration in m³.
    pub infiltration_volume_m3: f64,
    /// Capped evaporation in m³.
    pub evaporation_volume_m3: f64,
    /// Explicit numerical soil water correction in m³.
    pub soil_water_adjustment_volume_m3: f64,
    /// Full conservative drainage export in m³.
    pub export_volume_m3: f64,
    /// Complementary direct outlet export in m³.
    pub outlet_volume_m3: f64,
    /// Authoritative routed donor volume in m³.
    pub routed_volume_m3: f64,
    /// Endpoint discharge in m³/s. None denotes instantaneous overflow.
    pub endpoint_discharge_m3_per_s: Option<f64>,
    /// Independent subarea water residual in m³.
    pub water_residual_m3: f64,
}

/// Complete supported-class and outlet output of one catchment.
#[derive(Clone, Debug, PartialEq)]
pub struct SubcatchmentOutput {
    /// Ordered catchment index.
    pub catchment: usize,
    /// Fixed ordered classes. None denotes physically absent support.
    pub subareas: [Option<SubareaOutput>; 3],
    /// Total accepted represented storage in m³.
    pub storage_m3: f64,
    /// Conservative outlet volume over the accepted interval, in m³.
    pub outlet_volume_m3: f64,
    /// Sum of class endpoint discharges, undefined when any represented class is instantaneous.
    pub endpoint_discharge_m3_per_s: Option<f64>,
    /// Independent whole-catchment water residual in m³.
    pub water_residual_m3: f64,
    /// Applied-minus-total-area rain diagnostic in m³.
    pub rain_area_difference_m3: f64,
}

pub(super) fn outputs(
    receipts: Vec<SubcatchmentReceipt>,
) -> Result<Vec<SubcatchmentOutput>, SubcatchmentError> {
    receipts
        .into_iter()
        .map(|receipt| {
            let subareas = receipt.subareas.map(|r| {
                r.map(|r| SubareaOutput {
                    depth_m: r.drainage.depth_m,
                    storage_m3: r.drainage.storage_after_m3,
                    rain_volume_m3: r.rain_volume_m3,
                    external_runon_volume_m3: r.external_volume_m3,
                    internal_received_volume_m3: r.internal_received_volume_m3,
                    infiltration_volume_m3: r.infiltration_volume_m3,
                    evaporation_volume_m3: r.evaporation_volume_m3,
                    soil_water_adjustment_volume_m3: r.adjustment_volume_m3,
                    export_volume_m3: r.drainage.export_volume_m3,
                    outlet_volume_m3: r.partition.outlet_volume_m3,
                    routed_volume_m3: r.partition.routed_volume_m3,
                    endpoint_discharge_m3_per_s: r.drainage.endpoint_discharge_m3_per_s,
                    water_residual_m3: r.water_residual_m3,
                })
            });
            let storage = sum(subareas.iter().flatten().map(|s| s.storage_m3))?;
            let outlet = sum(subareas.iter().flatten().map(|s| s.outlet_volume_m3))?;
            let discharge = if subareas
                .iter()
                .flatten()
                .any(|s| s.endpoint_discharge_m3_per_s.is_none())
            {
                None
            } else {
                Some(sum(subareas
                    .iter()
                    .flatten()
                    .filter_map(|s| s.endpoint_discharge_m3_per_s))?)
            };
            Ok(SubcatchmentOutput {
                catchment: receipt.catchment,
                subareas,
                storage_m3: storage,
                outlet_volume_m3: outlet,
                endpoint_discharge_m3_per_s: discharge,
                water_residual_m3: receipt.water_residual_m3,
                rain_area_difference_m3: receipt.rain_area_difference_m3,
            })
        })
        .collect()
}
fn sum(values: impl Iterator<Item = f64>) -> Result<f64, SubcatchmentError> {
    let sum = values.fold(0.0, |a, b| a + b);
    if !sum.is_finite() {
        return Err(SubcatchmentError::numerical(
            "requested catchment output aggregate overflows. Rescale physical support",
        ));
    }
    Ok(sum)
}
