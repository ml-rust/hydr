//! Surface adapter for the shared native/SI soil receipt checks.

use super::check::rejected;
use crate::contract::EngineError;
use crate::solver::{SoilSourceReceipt, SourceReceipt};
use hydr_soil::internal::{SoilReceiptInput, check_source_receipt};

pub(super) fn check(
    receipt: &SourceReceipt,
    soil: &SoilSourceReceipt,
    rain: f64,
    available: f64,
    dt: f64,
) -> Result<(), EngineError> {
    check_source_receipt(
        &SoilReceiptInput {
            dt_s: dt,
            incoming_rate_m_per_s: rain,
            ponded_depth_m: receipt.ponded_depth_m,
            incoming_depth_m: receipt.rain_depth_m,
            available_depth_m: available,
            infiltration_depth_m: receipt.infiltration_depth_m,
            adjustment_depth_m: receipt.water_adjustment_depth_m,
        },
        soil,
    )
    .map_err(|error| {
        rejected(
            "soil source receipt",
            format!("cell={}: {error}", receipt.cell),
        )
    })
}
