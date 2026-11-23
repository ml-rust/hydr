//! Certified coefficient bounds of the shallow series, one column per barrel row.

use super::context::Bounds;
use crate::structures::numerics::culvert::constants;
use numr::dtype::DType;
use numr::runtime::Runtime;
use numr::tensor::Tensor;

/// Coefficient bounds of the three shallow series, each `[m]` wide.
pub(crate) struct SeriesTensors<R: Runtime<DType = DType>> {
    pub(crate) asin: Vec<Bounds<R>>,
    pub(crate) area: Vec<Bounds<R>>,
    pub(crate) moment: Vec<Bounds<R>>,
}

impl<R: Runtime<DType = DType>> SeriesTensors<R> {
    /// The certified series coefficient bounds, each `[m]` wide on `device`.
    pub(crate) fn constant(m: usize, device: &R::Device) -> numr::error::Result<Self> {
        let bounds = |table: &[[u64; 2]]| -> numr::error::Result<Vec<Bounds<R>>> {
            table
                .iter()
                .map(|[lower, upper]| {
                    Ok([
                        Tensor::<R>::full_scalar(&[m], DType::F64, f64::from_bits(*lower), device)?,
                        Tensor::<R>::full_scalar(&[m], DType::F64, f64::from_bits(*upper), device)?,
                    ])
                })
                .collect()
        };
        Ok(Self {
            asin: bounds(&constants::ASIN)?,
            area: bounds(&constants::AREA)?,
            moment: bounds(&constants::MOMENT)?,
        })
    }
}
