//! Compact `[m]` barrel rows: shape mask, dimensions, series coefficients and full-state bounds.

use crate::structures::numerics::culvert::{BarrelClient, constants, section};
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::runtime::{Runtime, RuntimeClient, SquareRootContract};
use numr::tensor::Tensor;
use solvr::common::arithmetic::with_interval_arithmetic;

/// Lower and upper endpoint tensors of one stored enclosure.
pub(crate) type Bounds<R> = [Tensor<R>; 2];

/// Coefficient bounds of the three shallow series, each `[m]` wide.
pub(crate) struct SeriesTensors<R: Runtime<DType = DType>> {
    pub(crate) asin: Vec<Bounds<R>>,
    pub(crate) area: Vec<Bounds<R>>,
    pub(crate) moment: Vec<Bounds<R>>,
}

/// One row per barrel. Circle and box rows mix and the `circular` mask selects per row.
pub(crate) struct TensorBarrelContext<R: Runtime<DType = DType>> {
    pub(crate) barrels: Vec<CulvertBarrel>,
    pub(crate) client: R::Client,
    pub(crate) device: R::Device,
    pub(crate) zero: Tensor<R>,
    pub(crate) one: Tensor<R>,
    /// 1 on circle rows, 0 on box rows.
    pub(crate) circular: Tensor<R>,
    /// 1 on box rows, 0 on circle rows.
    pub(crate) boxed: Tensor<R>,
    /// Rise `D`, the diameter of a circle.
    pub(crate) rise: Tensor<R>,
    /// Span `B` of a box, `D` for a circle.
    pub(crate) span: Tensor<R>,
    /// `D / 2`, exact.
    pub(crate) half_rise: Tensor<R>,
    /// `D / 16`, exact. Circle depths below it take the shallow series.
    pub(crate) shallow_top: Tensor<R>,
    /// `D / 32`, exact. Harmless shallow operand on other lanes.
    pub(crate) shallow_operand: Tensor<R>,
    pub(crate) series: SeriesTensors<R>,
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_perimeter: Bounds<R>,
    pub(crate) full_radius: Bounds<R>,
}

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Checks the runtime, uploads the rows and encloses the full-state constants.
    pub(crate) fn new(
        client: R::Client,
        barrels: Vec<CulvertBarrel>,
    ) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        if barrels.is_empty() {
            return Err(StructureError::definition(
                "barrels=[]: supply at least one culvert barrel",
            ));
        }
        if let Some(error) = crate::tensor::probe_f64::<R, _>(&client)? {
            return Err(StructureError::definition(error.message("culvert barrel")));
        }
        if let Some(error) = crate::tensor::probe_correctly_rounded::<R>() {
            return Err(StructureError::definition(error.message("culvert barrel")));
        }
        let sqrt = R::square_root_contract(DType::F64);
        if sqrt != SquareRootContract::Ieee754RoundToNearest {
            return Err(StructureError::definition(format!(
                "unsupported {} square root: contract={sqrt:?}. Select a runtime with F64 square roots rounded to nearest before creating the culvert barrel context",
                R::name()
            )));
        }
        let device = client.device().clone();
        let m = barrels.len();
        let column = |f: &dyn Fn(&CulvertBarrel) -> f64| -> numr::error::Result<Tensor<R>> {
            let values: Vec<f64> = barrels.iter().map(f).collect();
            Tensor::<R>::from_slice(&values, &[m], &device)
        };
        let circle = |b: &CulvertBarrel| b.shape() == CulvertBarrelShape::Circular;
        let circular = column(&|b| if circle(b) { 1.0 } else { 0.0 })?;
        let boxed = column(&|b| if circle(b) { 0.0 } else { 1.0 })?;
        let rise = column(&|b| b.rise_m())?;
        let span = column(&|b| b.span_m())?;
        let half_rise = column(&|b| b.rise_m() * 0.5)?;
        let shallow_top = column(&|b| b.rise_m() * 0.0625)?;
        let shallow_operand = column(&|b| b.rise_m() * 0.03125)?;
        let bounds = |table: &[[u64; 2]]| -> numr::error::Result<Vec<Bounds<R>>> {
            table
                .iter()
                .map(|[lower, upper]| {
                    Ok([
                        Tensor::<R>::full_scalar(
                            &[m],
                            DType::F64,
                            f64::from_bits(*lower),
                            &device,
                        )?,
                        Tensor::<R>::full_scalar(
                            &[m],
                            DType::F64,
                            f64::from_bits(*upper),
                            &device,
                        )?,
                    ])
                })
                .collect()
        };
        let series = SeriesTensors {
            asin: bounds(&constants::ASIN)?,
            area: bounds(&constants::AREA)?,
            moment: bounds(&constants::MOMENT)?,
        };
        let [full_area, full_perimeter, full_radius] = with_interval_arithmetic(
            &client,
            DType::F64,
            "culvert barrel full-state constants",
            |a| section::full_constants(a, &circular, &span, &rise),
        )?;
        Ok(Self {
            barrels,
            zero: Tensor::<R>::zeros(&[m], DType::F64, &device)?,
            one: Tensor::<R>::ones(&[m], DType::F64, &device)?,
            client,
            device,
            circular,
            boxed,
            rise,
            span,
            half_rise,
            shallow_top,
            shallow_operand,
            series,
            full_area,
            full_perimeter,
            full_radius,
        })
    }

    /// Barrel count `m`.
    pub(crate) fn width(&self) -> usize {
        self.barrels.len()
    }

    /// `[m]` tensor holding `value` in every row.
    pub(crate) fn constant(&self, value: f64) -> numr::error::Result<Tensor<R>> {
        Tensor::<R>::full_scalar(&[self.width()], DType::F64, value, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::StructureInputErrorKind;
    use numr::runtime::cpu::{CpuDevice, CpuRuntime};

    #[test]
    fn empty_barrel_list_is_rejected_with_action() {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let error = TensorBarrelContext::<CpuRuntime>::new(client, Vec::new()).err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { kind: StructureInputErrorKind::Definition, ref context, .. })
                if context.contains("supply at least one culvert barrel")
        ));
    }

    #[test]
    fn rows_hold_exact_fractions_of_the_rise() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = vec![
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
        ];
        let context = TensorBarrelContext::<CpuRuntime>::new(client, barrels)?;
        assert_eq!(context.circular.try_to_vec::<f64>()?, [1.0, 0.0]);
        assert_eq!(context.span.try_to_vec::<f64>()?, [0.3, 2.0]);
        assert_eq!(
            context.shallow_top.try_to_vec::<f64>()?,
            [0.3 / 16.0, 0.0625]
        );
        assert_eq!(context.series.asin.len(), constants::SERIES_TERMS + 1);
        // Box area B D = 2 and perimeter 2 (B + D) = 6 are exact singletons.
        for (bounds, expected) in [(&context.full_area, 2.0), (&context.full_perimeter, 6.0)] {
            for endpoint in bounds {
                assert_eq!(endpoint.try_to_vec::<f64>()?[1], expected);
            }
        }
        Ok(())
    }
}
