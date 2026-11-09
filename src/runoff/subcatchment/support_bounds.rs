//! Constructor-only outward propagation for represented subarea support.

use super::{SubcatchmentError, SubcatchmentGeometry};
use crate::soil::SoilError;
use crate::soil::admission::{rounding_bound, upward_add, upward_mul};

fn product(a: f64, ea: f64, b: f64, eb: f64) -> Result<f64, SoilError> {
    let inherited = upward_add(
        upward_add(upward_mul(a.abs(), eb)?, upward_mul(b.abs(), ea)?)?,
        upward_mul(ea, eb)?,
    )?;
    upward_add(inherited, rounding_bound(a * b)?)
}

pub(super) fn area_bounds(
    g: SubcatchmentGeometry,
    areas: [f64; 3],
) -> Result<[f64; 3], SubcatchmentError> {
    let derive = || -> Result<[f64; 3], SoilError> {
        let ai = g.total_area_m2 * g.impervious_fraction;
        let eai = product(g.total_area_m2, 0.0, g.impervious_fraction, 0.0)?;
        let complement_zero = 1.0 - g.zero_storage_impervious_fraction;
        let complement_pervious = 1.0 - g.impervious_fraction;
        Ok([
            product(ai, eai, g.zero_storage_impervious_fraction, 0.0)?,
            product(ai, eai, complement_zero, rounding_bound(complement_zero)?)?,
            product(
                g.total_area_m2,
                0.0,
                complement_pervious,
                rounding_bound(complement_pervious)?,
            )?,
        ])
    };
    let mut bounds = derive().map_err(|e| SubcatchmentError::input(e.to_string()))?;
    // Rejected underflow cannot reach here. Absent fractions have exact mathematical zero support.
    for (area, bound) in areas.iter().zip(&mut bounds) {
        if *area == 0.0 {
            *bound = 0.0;
        }
    }
    Ok(bounds)
}
