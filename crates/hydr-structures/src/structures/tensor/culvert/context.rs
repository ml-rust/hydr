//! Compact `[m]` barrel rows: shape masks, dimensions, series coefficients, arc layouts and
//! full-state bounds.

use super::arcs::ArcRows;
use crate::structures::numerics::culvert::arcs::ArcInputs;
use crate::structures::numerics::culvert::{BarrelClient, constants, section};
use crate::structures::{CulvertBarrel, CulvertBarrelShape, StructureError};
use numr::dtype::DType;
use numr::ops::{ConditionalOps, IndexingOps};
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

/// One row per barrel. Circle, box and arc rows mix and the shape masks select per row.
///
/// Every field lives on the device, so [`TensorBarrelContext::select`] gathers rows without a
/// host read. The owner of the context keeps the host barrel list.
pub(crate) struct TensorBarrelContext<R: Runtime<DType = DType>> {
    pub(crate) client: R::Client,
    pub(crate) device: R::Device,
    pub(crate) zero: Tensor<R>,
    pub(crate) one: Tensor<R>,
    /// 1 on circle rows, 0 on other rows.
    pub(crate) circular: Tensor<R>,
    /// 1 on box rows, 0 on other rows.
    pub(crate) boxed: Tensor<R>,
    /// Rise `D`, the diameter of a circle.
    pub(crate) rise: Tensor<R>,
    /// Span `B`, `D` for a circle and the floor chord for a flat-floor arch.
    pub(crate) span: Tensor<R>,
    /// `D / 2`, exact.
    pub(crate) half_rise: Tensor<R>,
    /// `D / 16`, exact. Circle depths below it take the shallow series.
    pub(crate) shallow_top: Tensor<R>,
    /// `D / 32`, exact. Harmless shallow operand on other lanes.
    pub(crate) shallow_operand: Tensor<R>,
    pub(crate) series: SeriesTensors<R>,
    /// Upper end of the critical-depth domain in `s = y/D`: `1 - 2^-53` circle, `1` box.
    /// NaN on arc rows, which hold no depth certificate yet.
    pub(crate) critical_top: Tensor<R>,
    /// Lower bound of `d r_c / ds` on the critical domain: `5.4` circle, `3` box, NaN arc.
    pub(crate) critical_slope: Tensor<R>,
    /// Upper end of the normal-depth domain in `s = y/D`: `s_N = 0.93818` circle, `1` box,
    /// NaN arc.
    pub(crate) normal_top: Tensor<R>,
    /// Lower bound of `d r_n / ds` on the normal domain: `5.42e-5` circle, `3` box, NaN arc.
    pub(crate) normal_slope: Tensor<R>,
    pub(crate) full_area: Bounds<R>,
    pub(crate) full_perimeter: Bounds<R>,
    pub(crate) full_radius: Bounds<R>,
    /// Arc layouts, with a harmless arc barrel on circle and box rows. `None` when no row is an
    /// arc barrel, so circle and box queries skip the arc branch.
    pub(crate) arcs: Option<ArcRows<R>>,
}

/// Rejects the first arc barrel of `barrels` for `query`, which arc rows do not serve yet.
///
/// The error names the barrel's row as its element.
pub(crate) fn reject_arc_barrels(
    barrels: &[CulvertBarrel],
    query: &str,
) -> Result<(), StructureError> {
    match barrels.iter().enumerate().find(|(_, b)| b.shape().arced()) {
        None => Ok(()),
        Some((index, barrel)) => Err(StructureError::definition(format!(
            "shape={:?},span_m={:?},rise_m={:?}: {query} on arc barrels not yet available. Use a circular or box barrel",
            barrel.shape(),
            barrel.span_m(),
            barrel.rise_m()
        ))
        .at(index)),
    }
}

impl<R: Runtime<DType = DType>> TensorBarrelContext<R> {
    /// Checks the runtime, uploads the rows and encloses the full-state constants.
    pub(crate) fn new(client: R::Client, barrels: &[CulvertBarrel]) -> Result<Self, StructureError>
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
        Self::build(client, barrels)
    }

    /// A context of unit-diameter circles with this context's width and client.
    ///
    /// The critical and normal residuals read circle geometry in `s = y/D` from it, so tiny rises never
    /// underflow the area.
    pub(crate) fn unit_circles(&self) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let unit = CulvertBarrel::circular(1.0)?;
        Self::build(self.client.clone(), &vec![unit; self.width()])
    }

    /// Gathers the rows at the I64 indices `rows`, in their order. Indices may repeat.
    pub(crate) fn select(&self, rows: &Tensor<R>) -> numr::error::Result<Self>
    where
        R::Client: BarrelClient<R>,
    {
        let c = &self.client;
        let take = |value: &Tensor<R>| c.index_select(value, 0, rows);
        let pair = |[lower, upper]: &Bounds<R>| -> numr::error::Result<Bounds<R>> {
            Ok([take(lower)?, take(upper)?])
        };
        let table = |values: &[Bounds<R>]| -> numr::error::Result<Vec<Bounds<R>>> {
            values.iter().map(pair).collect()
        };
        Ok(Self {
            client: c.clone(),
            device: self.device.clone(),
            zero: take(&self.zero)?,
            one: take(&self.one)?,
            circular: take(&self.circular)?,
            boxed: take(&self.boxed)?,
            rise: take(&self.rise)?,
            span: take(&self.span)?,
            half_rise: take(&self.half_rise)?,
            shallow_top: take(&self.shallow_top)?,
            shallow_operand: take(&self.shallow_operand)?,
            series: SeriesTensors {
                asin: table(&self.series.asin)?,
                area: table(&self.series.area)?,
                moment: table(&self.series.moment)?,
            },
            critical_top: take(&self.critical_top)?,
            critical_slope: take(&self.critical_slope)?,
            normal_top: take(&self.normal_top)?,
            normal_slope: take(&self.normal_slope)?,
            full_area: pair(&self.full_area)?,
            full_perimeter: pair(&self.full_perimeter)?,
            full_radius: pair(&self.full_radius)?,
            arcs: self
                .arcs
                .as_ref()
                .map(|arcs| arcs.select(&take))
                .transpose()?,
        })
    }

    /// Uploads the rows of an admitted runtime and encloses the full-state constants.
    fn build(client: R::Client, barrels: &[CulvertBarrel]) -> Result<Self, StructureError>
    where
        R::Client: BarrelClient<R>,
    {
        let device = client.device().clone();
        let m = barrels.len();
        let column = |f: &dyn Fn(&CulvertBarrel) -> f64| -> numr::error::Result<Tensor<R>> {
            let values: Vec<f64> = barrels.iter().map(f).collect();
            Tensor::<R>::from_slice(&values, &[m], &device)
        };
        let circle = |b: &CulvertBarrel| b.shape() == CulvertBarrelShape::Circular;
        let flag = |value: bool| f64::from(u8::from(value));
        let circular = column(&|b| flag(circle(b)))?;
        let boxed = column(&|b| flag(b.shape() == CulvertBarrelShape::Box))?;
        let arc_rows = column(&|b| flag(b.shape().arced()))?;
        let rise = column(&|b| b.rise_m())?;
        let span = column(&|b| b.span_m())?;
        let half_rise = column(&|b| b.rise_m() * 0.5)?;
        let shallow_top = column(&|b| b.rise_m() * 0.0625)?;
        let shallow_operand = column(&|b| b.rise_m() * 0.03125)?;
        // Circle value, box value. Arc rows hold NaN until their certificates exist.
        let certified = |[on_circle, on_box]: [f64; 2]| {
            column(&|b| match b.shape() {
                CulvertBarrelShape::Circular => on_circle,
                CulvertBarrelShape::Box => on_box,
                _ => f64::NAN,
            })
        };
        let critical_top = certified([constants::CIRCLE_CRITICAL_TOP, 1.0])?;
        let critical_slope = certified([
            constants::CIRCLE_CRITICAL_SLOPE,
            constants::BOX_CRITICAL_SLOPE,
        ])?;
        let normal_top = certified([constants::RISING_LIMB_TOP, 1.0])?;
        let normal_slope =
            certified([constants::CIRCLE_NORMAL_SLOPE, constants::BOX_NORMAL_SLOPE])?;
        let series = SeriesTensors::constant(m, &device)?;
        let inputs = if barrels.iter().any(|b| b.shape().arced()) {
            Some(ArcInputs::<R>::new(barrels, &device)?)
        } else {
            None
        };
        let (closed, arcs) = with_interval_arithmetic(
            &client,
            DType::F64,
            "culvert barrel full-state constants",
            |a| {
                let closed = section::full_constants(a, &circular, &span, &rise)?;
                let arcs = match &inputs {
                    Some(inputs) => Some(ArcRows::build(a, inputs, &arc_rows, &span, &rise)?),
                    None => None,
                };
                Ok((closed, arcs))
            },
        )?;
        let merge = |closed: Bounds<R>,
                     arc: fn(&ArcRows<R>) -> &Bounds<R>|
         -> numr::error::Result<Bounds<R>> {
            let Some(arcs) = &arcs else {
                return Ok(closed);
            };
            let [lower, upper] = &closed;
            let [arc_lower, arc_upper] = arc(arcs);
            Ok([
                client.where_cond(&arc_rows, arc_lower, lower)?,
                client.where_cond(&arc_rows, arc_upper, upper)?,
            ])
        };
        let [area, perimeter, radius] = closed;
        let full_area = merge(area, |arcs| &arcs.full_area)?;
        let full_perimeter = merge(perimeter, |arcs| &arcs.full_perimeter)?;
        let full_radius = merge(radius, |arcs| &arcs.full_radius)?;
        Ok(Self {
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
            critical_top,
            critical_slope,
            normal_top,
            normal_slope,
            full_area,
            full_perimeter,
            full_radius,
            arcs,
        })
    }

    /// Barrel count `m`.
    pub(crate) fn width(&self) -> usize {
        self.rise.shape()[0]
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
        let error = TensorBarrelContext::<CpuRuntime>::new(client, &[]).err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { kind: StructureInputErrorKind::Definition, ref context, .. })
                if context.contains("supply at least one culvert barrel")
        ));
    }

    #[test]
    fn arc_rows_join_the_context_and_reject_depth_queries_by_row() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = [
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::pipe_arch(1.8542, 1.397, 0.4572, 1.938_528)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
            CulvertBarrel::arch(3.048, 1.6002)?,
        ];
        let context = TensorBarrelContext::<CpuRuntime>::new(client.clone(), &barrels)?;
        assert_eq!(context.boxed.try_to_vec::<f64>()?, [0.0, 0.0, 1.0, 0.0]);
        let arcs = context
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("context lost its arc rows"))?;
        assert_eq!(arcs.arced.try_to_vec::<f64>()?, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(
            arcs.floor_width.try_to_vec::<f64>()?,
            [0.0, 0.0, 0.0, 3.048]
        );
        let slopes = context.critical_slope.try_to_vec::<f64>()?;
        assert!(slopes[1].is_nan() && slopes[3].is_nan() && slopes[0] == 5.4);
        let rows = Tensor::<CpuRuntime>::from_slice(&[3i64, 2], &[2], &device)?;
        let picked = context.select(&rows)?;
        let picked_arcs = picked
            .arcs
            .as_ref()
            .ok_or_else(|| StructureError::definition("selection lost its arc rows"))?;
        assert_eq!(picked_arcs.floor_width.try_to_vec::<f64>()?, [3.048, 0.0]);
        let circles = TensorBarrelContext::<CpuRuntime>::new(client, &barrels[..1])?;
        assert!(circles.arcs.is_none());

        let error = reject_arc_barrels(&barrels, "critical depth").err();
        assert!(matches!(
            error,
            Some(StructureError::InvalidInput { element: Some(1), kind: StructureInputErrorKind::Definition, ref context })
                if context == "shape=PipeArch,span_m=1.8542,rise_m=1.397: critical depth on arc barrels not yet available. Use a circular or box barrel"
        ));
        assert!(reject_arc_barrels(&barrels[..1], "critical depth").is_ok());
        Ok(())
    }

    #[test]
    fn rows_hold_exact_fractions_of_the_rise() -> Result<(), StructureError> {
        let device = CpuDevice::new();
        let client = CpuRuntime::default_client(&device);
        let barrels = vec![
            CulvertBarrel::circular(0.3)?,
            CulvertBarrel::rectangular_box(2.0, 1.0)?,
        ];
        let context = TensorBarrelContext::<CpuRuntime>::new(client, &barrels)?;
        assert_eq!(context.circular.try_to_vec::<f64>()?, [1.0, 0.0]);
        assert_eq!(context.span.try_to_vec::<f64>()?, [0.3, 2.0]);
        assert_eq!(
            context.shallow_top.try_to_vec::<f64>()?,
            [0.3 / 16.0, 0.0625]
        );
        assert_eq!(context.series.asin.len(), constants::SERIES_TERMS + 1);
        let rows = Tensor::<CpuRuntime>::from_slice(&[1i64, 0, 1], &[3], &device)?;
        let picked = context.select(&rows)?;
        assert_eq!(picked.width(), 3);
        assert_eq!(picked.rise.try_to_vec::<f64>()?, [1.0, 0.3, 1.0]);
        assert_eq!(picked.critical_slope.try_to_vec::<f64>()?, [3.0, 5.4, 3.0]);
        assert_eq!(picked.normal_top.try_to_vec::<f64>()?, [1.0, 0.93818, 1.0]);
        assert_eq!(
            picked.normal_slope.try_to_vec::<f64>()?,
            [3.0, 5.42e-5, 3.0]
        );
        assert_eq!(picked.full_perimeter[1].try_to_vec::<f64>()?[2], 6.0);
        assert_eq!(picked.series.moment[3][0].shape(), [3]);
        // Box area B D = 2 and perimeter 2 (B + D) = 6 are exact singletons.
        for (bounds, expected) in [(&context.full_area, 2.0), (&context.full_perimeter, 6.0)] {
            for endpoint in bounds {
                assert_eq!(endpoint.try_to_vec::<f64>()?[1], expected);
            }
        }
        Ok(())
    }
}
